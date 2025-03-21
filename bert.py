import os
import requests
import json
import re
import torch
from transformers import BertTokenizer, BertModel
from dotenv import load_dotenv
import spacy
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

load_dotenv()
api_key = os.getenv("API_KEY")

# Constants
NON_FOOD_WORDS = [
    'and', 'with', 'the', 'of', 'for', 'a', 'an', 'to', 'on', 'at', 'in', 'by', 
    'or', 'as', 'if', 'so', 'but', 'from', 'than', 'had', 'has', 'have', 'is', 
    'are', 'was', 'were', 'be', 'been', 'being', 'that', 'this', 'these', 'those',
    'it', 'its', 'they', 'them', 'their', 'theirs', 'he', 'him', 'his', 'she', 
    'her', 'hers', 'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 
    'our', 'ours'
]

COMMON_FOODS = [
    "egg", "eggs", "toast", "bread", "butter", "peanut butter", "jam", "cereal", 
    "milk", "cheese", "yogurt", "chicken", "beef", "pork", "fish", "apple", 
    "banana", "orange", "rice", "pasta", "potato", "tomato", "carrot", "onion",
    "lettuce", "salad", "soup", "sandwich", "pizza", "burger", "fries", "chocolate",
    "cookie", "cake", "ice cream", "coffee", "tea", "juice", "water", "soda"
]

MEAL_TIMES = ["breakfast", "lunch", "dinner", "snack", "morning", "afternoon", "evening"]

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Load a minimal spaCy model for tokenization and basic linguistic features
nlp = spacy.load("en_core_web_md")

# Custom component to identify food items using BERT embeddings
class BertFoodIdentifier:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.food_embeddings = self._get_food_embeddings()
        
    def _get_food_embeddings(self):
        # Create embeddings for common foods
        embeddings = {}
        for food in COMMON_FOODS:
            inputs = tokenizer(food, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings[food] = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embeddings
    
    def get_embedding(self, text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def cosine_similarity(self, a, b):
        return torch.nn.functional.cosine_similarity(
            torch.tensor(a).unsqueeze(0), 
            torch.tensor(b).unsqueeze(0)
        ).item()
    
    def is_food(self, text):
        if text.lower() in COMMON_FOODS:
            return True
            
        # Check using BERT embeddings
        text_embedding = self.get_embedding(text)
        for food, embedding in self.food_embeddings.items():
            similarity = self.cosine_similarity(text_embedding, embedding)
            if similarity > self.threshold:
                return True
                
        return False

# Initialize the BERT food identifier
bert_food_identifier = BertFoodIdentifier()

def identify_quantities_and_mealtimes(doc):
    """Identify quantities and meal times in the document"""
    entities = []
    
    # Look for numbers followed by units
    quantity_patterns = [
        (r'\d+(\.\d+)?\s*(gr|gram|grams|g)\b', "QUANTITY"),
        (r'\d+(\.\d+)?\s*(oz|ounce|ounces)\b', "QUANTITY"),
        (r'\d+(\.\d+)?\s*(cup|cups)\b', "QUANTITY"),
        (r'\d+(\.\d+)?\s*(tbsp|tablespoon|tablespoons)\b', "QUANTITY"),
        (r'\d+(\.\d+)?\s*(tsp|teaspoon|teaspoons)\b', "QUANTITY"),
        (r'\d+(\.\d+)?\s*(slice|slices|piece|pieces)\b', "QUANTITY"),
        (r'\d+(\.\d+)?\s*(box|boxes|package|packages)\b', "QUANTITY"),
        (r'\d+(\.\d+)?\b', "QUANTITY")  # Just numbers
    ]
    
    text = doc.text
    for pattern, label in quantity_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start_char = match.start()
            end_char = match.end()
            
            # Find token spans
            start_token = None
            end_token = None
            for i, token in enumerate(doc):
                if token.idx <= start_char < token.idx + len(token.text):
                    start_token = i
                if token.idx <= end_char <= token.idx + len(token.text):
                    end_token = i + 1
                    break
            
            if start_token is not None and end_token is not None:
                entities.append((start_token, end_token, label))
    
    # Look for meal times
    for meal_time in MEAL_TIMES:
        for match in re.finditer(r'\b' + meal_time + r'\b', text, re.IGNORECASE):
            start_char = match.start()
            end_char = match.end()
            
            # Find token spans
            start_token = None
            end_token = None
            for i, token in enumerate(doc):
                if token.idx <= start_char < token.idx + len(token.text):
                    start_token = i
                if token.idx <= end_char <= token.idx + len(token.text):
                    end_token = i + 1
                    break
            
            if start_token is not None and end_token is not None:
                entities.append((start_token, end_token, "MEAL_TIME"))
    
    return entities

def identify_food_items_bert(doc):
    """Identify food items using BERT and rules"""
    entities = []
    
    # Check each noun chunk for food items
    for chunk in doc.noun_chunks:
        if (chunk.text.lower() not in NON_FOOD_WORDS and
            not all(token.is_stop for token in chunk)):
            if bert_food_identifier.is_food(chunk.text):
                entities.append((chunk.start, chunk.end, "POTENTIAL_FOOD"))
    
    # Check individual nouns that might be foods
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in NON_FOOD_WORDS:
            if bert_food_identifier.is_food(token.text):
                # Check if this token is already part of an entity
                is_part_of_entity = False
                for start, end, _ in entities:
                    if start <= token.i < end:
                        is_part_of_entity = True
                        break
                if not is_part_of_entity:
                    entities.append((token.i, token.i + 1, "POTENTIAL_FOOD"))
    
    return entities

def process_text_with_bert(text):
    """Process food log text using BERT and spaCy"""
    # Process with spaCy for tokenization and basic linguistic features
    doc = nlp(text)
    quantity_meal_entities = identify_quantities_and_mealtimes(doc)
    food_entities = identify_food_items_bert(doc)
    
    # Combine all entities
    all_entities = quantity_meal_entities + food_entities
    spans = [Span(doc, start, end, label=label) for start, end, label in all_entities]
    filtered_spans = filter_spans(spans)
    
    # Update doc.ents
    doc.ents = filtered_spans
    return doc

# Function to query USDA API for food items
def query_usda_for_food(food_item):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"query": food_item, "api_key": api_key, "pageSize": 5}
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['foods']:
                match = data['foods'][0]
                return {
                    "description": match['description'],
                    "fdcId": match['fdcId'],
                    "nutrients": [n for n in match.get('foodNutrients', [])[:5]]
                }
            return {"description": f"No match found for {food_item}", "fdcId": None, "nutrients": []}
        else:
            print(f"API error: {response.status_code} for food item: {food_item}")
            return {"description": f"API error for {food_item}", "fdcId": None, "nutrients": []}
    except Exception as e:
        print(f"Exception: {str(e)} for food item: {food_item}")
        return {"description": f"Error processing {food_item}", "fdcId": None, "nutrients": []}

# Extract value from quantity string
def extract_quantity_value(qty_text):
    if not qty_text:
        return 1.0
    
    match = re.search(r'\d+(\.\d+)?', qty_text)
    return float(match.group()) if match else 1.0

# Process a single food log entry with BERT
def process_food_text(text):
    doc = process_text_with_bert(text)
    
    # Collect entities and their positions
    entities = {"food_items": [], "quantities": [], "meal_times": []}
    entity_positions = {}
    
    for ent in doc.ents:
        if ent.label_ == "POTENTIAL_FOOD":
            entities["food_items"].append(ent.text)
            entity_positions[ent.text] = {"type": "food", "start": ent.start, "end": ent.end}
        elif ent.label_ == "QUANTITY":
            entities["quantities"].append(ent.text)
            entity_positions[ent.text] = {"type": "quantity", "start": ent.start, "end": ent.end}
        elif ent.label_ == "MEAL_TIME":
            entities["meal_times"].append(ent.text)
            entity_positions[ent.text] = {"type": "meal_time", "start": ent.start, "end": ent.end}
    
    # Query USDA API for each food item
    food_details = {}
    for food in entities["food_items"]:
        details = query_usda_for_food(food)
        food_details[food] = details
    
    # Initialize meal groups
    meal_groups = {meal: [] for meal in entities["meal_times"]}
    if not entities["meal_times"]:
        meal_groups["unspecified"] = []
    
    # Associate foods with meals and quantities
    for food in entities["food_items"]:
        # Find closest meal
        food_pos = entity_positions[food]["start"]
        closest_meal = None
        min_distance = float('inf')
        
        for meal in entities["meal_times"]:
            meal_pos = entity_positions[meal]["start"]
            distance = abs(food_pos - meal_pos)
            if distance < min_distance:
                min_distance = distance
                closest_meal = meal
        
        # Find associated quantities
        associated_quantities = [
            qty for qty in entities["quantities"]
            if 0 <= entity_positions[food]["start"] - entity_positions[qty]["end"] <= 3
        ]
        
        # Create food entry with details
        details = food_details[food]
        serializable_details = {
            "description": details.get("description", ""),
            "fdcId": details.get("fdcId", None),
            "nutrients": []
        }
        
        # Add nutrients
        if "nutrients" in details and isinstance(details["nutrients"], list):
            for nutrient in details["nutrients"][:5]:
                if isinstance(nutrient, dict):
                    serializable_details["nutrients"].append({
                        "name": nutrient.get("nutrientName", ""),
                        "value": nutrient.get("value", "N/A"),
                        "unit": nutrient.get("unitName", "")
                    })
        
        # Create the food entry
        food_entry = {
            "item": food,
            "quantity": associated_quantities if associated_quantities else ["1 serving"],
            "details": serializable_details
        }
        
        # Add to appropriate meal group
        if closest_meal:
            meal_groups[closest_meal].append(food_entry)
        else:
            meal_groups["unspecified"].append(food_entry)
    
    return {"meals": meal_groups}

# Calculate nutritional summary
def calculate_nutritional_summary(meals):
    summary = {}
    
    for meal_name, foods in meals.items():
        for food in foods:
            if "details" in food and "nutrients" in food["details"]:
                for nutrient in food["details"]["nutrients"]:
                    try:
                        nutrient_name = nutrient["name"]
                        value = float(nutrient["value"])
                        unit = nutrient["unit"]
                        
                        # Apply quantity multiplier if available
                        if food["quantity"]:
                            qty_value = extract_quantity_value(food["quantity"][0])
                        else:
                            qty_value = 1.0
                            
                        value *= qty_value
                        
                        # Add to summary
                        key = f"{nutrient_name} ({unit})"
                        summary[key] = summary.get(key, 0) + value
                    except (ValueError, TypeError, KeyError, IndexError) as e:
                        # Print error for debugging
                        print(f"Error processing nutrient: {nutrient}, {str(e)}")
                        continue
    
    # Round values
    return {k: round(v, 2) for k, v in summary.items()}

# Main function to process food logs
def process_food_log(input_logs):
    all_results = {
        "meals": {},
        "nutritional_summary": {}
    }
    
    # Process each log entry
    for log_entry in input_logs:
        results = process_food_text(log_entry)
        
        # Merge results
        for meal, items in results["meals"].items():
            if meal in all_results["meals"]:
                all_results["meals"][meal].extend(items)
            else:
                all_results["meals"][meal] = items
    
    # Calculate nutritional summary
    all_results["nutritional_summary"] = calculate_nutritional_summary(all_results["meals"])
    
    return all_results

# Process multiple days of food logs
def process_daily_food_logs(daily_logs):
    all_results = {}
    
    # Process each day separately
    for date, logs in daily_logs.items():
        all_results[date] = process_food_log(logs)
    
    return all_results

if __name__ == "__main__":
    input_log = [
        "For breakfast, I had 20 gram salmon and 2 slices of salad with roasted beef.",
        "For lunch, I ate a chicken sandwich with tomato.",
        "My dinner served with 3 egg and 2 slices of bacon.",
    ]
    
    results = process_food_log(input_log)
    for i, log in enumerate(input_log, 1):
        print(f"{i}. {log}")
    
    print("\nJSON Output:")
    print(json.dumps(results, indent=2))