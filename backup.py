import os
import spacy
import requests
import json
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from dotenv import load_dotenv

# Load the spaCy model & .env
nlp = spacy.load("en_core_web_md")
load_dotenv()

# API Key
api_key = os.getenv("API_KEY")

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

# Create the entity ruler for quantities and meal times
ruler = nlp.add_pipe("entity_ruler", before="ner")

# Define more generic quantity patterns
quantity_patterns = [
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["gr", "gram", "grams", "g"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["oz", "ounce", "ounces"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["cup", "cups"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["tbsp", "tablespoon", "tablespoons"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["tsp", "teaspoon", "teaspoons"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["slice", "slices", "piece", "pieces"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["box", "boxes", "package", "packages"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["pcs", "piece", "pieces"]}}]},
    {"label": "QUANTITY", "pattern": [{"LIKE_NUM": True}]},  # Just a general number pattern
]

# Define patterns for meal times
meal_time_patterns = [
    {"label": "MEAL_TIME", "pattern": "breakfast"},
    {"label": "MEAL_TIME", "pattern": "lunch"},
    {"label": "MEAL_TIME", "pattern": "dinner"},
    {"label": "MEAL_TIME", "pattern": "snack"},
    {"label": "MEAL_TIME", "pattern": "morning"},
    {"label": "MEAL_TIME", "pattern": "afternoon"},
    {"label": "MEAL_TIME", "pattern": "evening"},
]

# Add the patterns to the entity ruler
ruler.add_patterns(quantity_patterns + meal_time_patterns)

# Create a PhraseMatcher for basic food items
basic_food_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
food_patterns = [nlp.make_doc(food) for food in COMMON_FOODS]
basic_food_matcher.add("BASIC_FOOD", food_patterns)

# Custom component to identify potential food items
@spacy.Language.component("food_identifier")
def identify_food_items(doc):
    # Get existing entities
    original_ents = list(doc.ents)
    
    # Find basic food matches
    matches = basic_food_matcher(doc)
    food_spans = []
    for match_id, start, end in matches:
        span = spacy.tokens.Span(doc, start, end, label="POTENTIAL_FOOD")
        food_spans.append(span)
    
    # Consider noun phrases as potential food items
    for noun_chunk in doc.noun_chunks:
        # Skip chunks that are stop words or in non-food words
        if (noun_chunk.text.lower() not in NON_FOOD_WORDS and 
            not all(token.is_stop for token in noun_chunk)):
            # Check if this chunk is already part of another span
            is_part_of_span = False
            for span in food_spans + original_ents:
                if (span.start <= noun_chunk.start < span.end or 
                    span.start < noun_chunk.end <= span.end):
                    is_part_of_span = True
                    break
            
            if not is_part_of_span:
                food_spans.append(spacy.tokens.Span(doc, noun_chunk.start, noun_chunk.end, label="POTENTIAL_FOOD"))
    
    # Look for food items that follow quantities
    quantity_indexes = [ent.end for ent in original_ents if ent.label_ == "QUANTITY"]
    for i in quantity_indexes:
        if i < len(doc):
            # Check if the token after a quantity is a noun and not already an entity
            token = doc[i]
            if token.pos_ == "NOUN" and token.ent_type_ == "":
                # Extend to include compound nouns
                end = i + 1
                while end < len(doc) and (doc[end].dep_ == "compound" or doc[end].pos_ == "NOUN"):
                    end += 1
                
                if token.text.lower() not in NON_FOOD_WORDS:
                    span = spacy.tokens.Span(doc, i, end, label="POTENTIAL_FOOD")
                    food_spans.append(span)
    
    # Combine all spans and filter out overlaps
    all_spans = original_ents + food_spans
    filtered_spans = filter_spans(all_spans)
    
    # Set entities
    doc.ents = filtered_spans
    return doc

# Add the food identifier after entity recognition
nlp.add_pipe("food_identifier", after="ner")

# Function to query USDA API for identified food items
def query_usda_for_food(food_item):
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    
    # Parameters to query the API
    params = {
        "query": food_item,
        "api_key": api_key,
        "pageSize": 5,  # Limit to top 5 matches
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data['foods']:
                # Return the first match details
                match = data['foods'][0]
                return {
                    "description": match['description'],
                    "fdcId": match['fdcId'],
                    "nutrients": [n for n in match.get('foodNutrients', [])[:5]]  # Get first 5 nutrients
                }
            else:
                return {"description": f"No match found for {food_item}", "fdcId": None}
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}

# Process text function
def process_food_log(input_logs):
    """Process a list of food log entries and combine the results"""
    all_results = {
        "meals": {},
        "nutritional_summary": {}  # Add new field for nutritional summary
    }
    
    # Process each log entry separately
    for log_entry in input_logs:
        results = process_food_text(log_entry)
        
        # Merge the results
        for meal, items in results["meals"].items():
            if meal in all_results["meals"]:
                all_results["meals"][meal].extend(items)
            else:
                all_results["meals"][meal] = items
    
    # Calculate nutritional summary across all meals
    all_results["nutritional_summary"] = calculate_nutritional_summary(all_results["meals"])
    
    return all_results

def calculate_nutritional_summary(meals):
    """Calculate total nutritional intake across all meals"""
    summary = {}
    
    # Iterate through all meals and food items
    for meal_name, foods in meals.items():
        for food in foods:
            if "details" in food and "nutrients" in food["details"]:
                # Process each nutrient
                for nutrient in food["details"]["nutrients"]:
                    nutrient_name = nutrient["name"]
                    
                    # Try to convert value to float for calculation
                    try:
                        value = float(nutrient["value"])
                    except (ValueError, TypeError):
                        continue  # Skip if value can't be converted to float
                    
                    unit = nutrient["unit"]
                    
                    # Apply quantity multiplier if available
                    if food["quantity"]:
                        # This is simplified - in reality, you'd need to parse the quantity string
                        # and convert it appropriately based on the food and unit
                        try:
                            # Extract first number from quantity string
                            qty_text = food["quantity"][0]
                            # Find the first number in the string
                            import re
                            qty_match = re.search(r'\d+(\.\d+)?', qty_text)
                            if qty_match:
                                multiplier = float(qty_match.group())
                                value *= multiplier
                        except (IndexError, ValueError, AttributeError):
                            # If can't parse quantity, use the value as is
                            pass
                    
                    # Add to summary
                    key = f"{nutrient_name} ({unit})"
                    if key in summary:
                        summary[key] += value
                    else:
                        summary[key] = value
    
    # Round values for better readability
    for key in summary:
        summary[key] = round(summary[key], 2)
    
    return summary

def process_daily_food_logs(daily_logs):
    """Process food logs for multiple days
    
    Args:
        daily_logs: Dictionary with dates as keys and lists of food log entries as values
    
    Returns:
        Dictionary with dates as keys and processed food logs as values
    """
    all_results = {}
    
    # Process each day separately
    for date, logs in daily_logs.items():
        all_results[date] = process_food_log(logs)
    
    return all_results

def process_food_text(text):
    doc = nlp(text)
    
    # Collect entities
    food_items = []
    quantities = []
    meal_times = []
    
    # Track position of entities for later grouping
    entity_positions = {}
    
    for ent in doc.ents:
        if ent.label_ == "POTENTIAL_FOOD":
            food_items.append(ent.text)
            entity_positions[ent.text] = {"type": "food", "start": ent.start, "end": ent.end}
        elif ent.label_ == "QUANTITY":
            quantities.append(ent.text)
            entity_positions[ent.text] = {"type": "quantity", "start": ent.start, "end": ent.end}
        elif ent.label_ == "MEAL_TIME":
            meal_times.append(ent.text)
            entity_positions[ent.text] = {"type": "meal_time", "start": ent.start, "end": ent.end}
    
    # Query USDA API for each food item
    food_details = {}
    for food in food_items:
        food_details[food] = query_usda_for_food(food)
    
    # Group items by meal time
    meal_groups = {}
    
    # Initialize meal groups with detected meal times
    for meal in meal_times:
        meal_groups[meal] = []
    
    # If no meal time is detected, create a default "unspecified" group
    if not meal_times:
        meal_groups["unspecified"] = []
    
    # Associate food items with meal times based on proximity in text
    for food in food_items:
        food_pos = entity_positions[food]["start"]
        closest_meal = None
        min_distance = float('inf')
        
        for meal in meal_times:
            meal_pos = entity_positions[meal]["start"]
            distance = abs(food_pos - meal_pos)
            
            if distance < min_distance:
                min_distance = distance
                closest_meal = meal
        
        # Find quantities that might be associated with this food
        associated_quantities = []
        for quantity in quantities:
            quantity_end = entity_positions[quantity]["end"]
            food_start = entity_positions[food]["start"]
            
            # Check if quantity immediately precedes food or with 1-2 tokens gap
            if 0 <= food_start - quantity_end <= 3:
                associated_quantities.append(quantity)
        
        # Create a serializable version of the food details
        serializable_details = {}
        details = food_details[food]
        
        # Extract only the serializable parts of details
        serializable_details["description"] = details.get("description", "")
        serializable_details["fdcId"] = details.get("fdcId", None)
        
        # Create a serializable version of nutrients
        if "nutrients" in details and isinstance(details["nutrients"], list):
            serializable_details["nutrients"] = []
            for nutrient in details["nutrients"][:5]:  # Top 5 nutrients
                if isinstance(nutrient, dict):
                    serializable_details["nutrients"].append({
                        "name": nutrient.get("nutrientName", ""),
                        "value": nutrient.get("value", "N/A"),
                        "unit": nutrient.get("unitName", "")
                    })
        
        food_entry = {
            "item": food,
            "quantity": associated_quantities if associated_quantities else ["1 serving"],
            "details": serializable_details
        }
        
        # Add food to the closest meal group or to unspecified if no meal times detected
        if closest_meal:
            meal_groups[closest_meal].append(food_entry)
        else:
            meal_groups["unspecified"].append(food_entry)
    
    # Create the final result object (completely serializable)
    result = {
        "meals": meal_groups
    }
    
    return result

if __name__ == "__main__":
    daily_logs = {
        "2025-03-19": [
            "For breakfast, I had 200 gram salmon and 2 slices of salad with roasted beef.",
            "For lunch, I ate a chicken sandwich with tomato.",
            "My dinner served with 3 egg and 2 slices of bacon.",
        ],
        "2025-03-20": [
            "Breakfast included 2 eggs and toast with butter.",
            "For lunch I had a bowl of vegetable soup.",
            "Dinner was grilled chicken with 100g of rice and vegetables."
        ]
    }
    
    results = process_daily_food_logs(daily_logs)
    json_output = json.dumps(results, indent=2)

    print("\nJSON Output:")
    print(json_output)