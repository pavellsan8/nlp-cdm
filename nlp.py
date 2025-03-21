import os
import spacy
import requests
import json
import re
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from dotenv import load_dotenv

# Load the spaCy model & .env
nlp = spacy.load("en_core_web_md")
load_dotenv()

# API Key
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

def filter_spans(spans):
    """Filter a sequence of spans and keep only the longest non-overlapping ones."""
    sorted_spans = sorted(spans, key=lambda span: (span.end - span.start), reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for overlap
        if not any(token.i in seen_tokens for token in span):
            result.append(span)
            seen_tokens.update(token.i for token in span)
    return result

# Initialize NLP pipeline components
def setup_nlp_pipeline():
    # Create the entity ruler for quantities and meal times
    if "entity_ruler" in nlp.pipe_names:
        ruler = nlp.get_pipe("entity_ruler")
    else:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Define quantity patterns
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
    
    # Define meal time patterns
    # Define meal time patterns
    meal_time_patterns = [
        {"label": "MEAL_TIME", "pattern": "breakfast"},
        {"label": "MEAL_TIME", "pattern": "lunch"},
        {"label": "MEAL_TIME", "pattern": "dinner"},
        {"label": "MEAL_TIME", "pattern": "snack"},
        {"label": "MEAL_TIME", "pattern": "morning"},
        {"label": "MEAL_TIME", "pattern": "afternoon"},
        {"label": "MEAL_TIME", "pattern": "evening"},
        # Add time pattern matches
        {"label": "MEAL_TIME", "pattern": [{"SHAPE": "dd:dd"}]},  # e.g. 08:30
        {"label": "MEAL_TIME", "pattern": [{"SHAPE": "d:dd"}]},   # e.g. 8:30
        {"label": "MEAL_TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["am", "pm"]}}]},  # e.g. 8 am
        {"label": "MEAL_TIME", "pattern": [{"LIKE_NUM": True}, {"LOWER": ":"}, {"LIKE_NUM": True}, {"LOWER": {"IN": ["am", "pm"]}}]},  # e.g. 8:30 am
    ]
    
    # Add patterns to ruler
    ruler.add_patterns(quantity_patterns + meal_time_patterns)
    
    # Add custom food identifier component
    if "food_identifier" not in nlp.pipe_names:
        nlp.add_pipe("food_identifier", after="ner")
    
    return nlp

@spacy.Language.component("food_identifier")
def identify_food_items(doc):
    # Create PhraseMatcher for basic foods
    basic_food_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    food_patterns = [nlp.make_doc(food) for food in COMMON_FOODS]
    basic_food_matcher.add("BASIC_FOOD", food_patterns)
    
    # Get existing entities
    original_ents = list(doc.ents)
    
    # Find basic food matches
    matches = basic_food_matcher(doc)
    food_spans = []
    for match_id, start, end in matches:
        span = spacy.tokens.Span(doc, start, end, label="POTENTIAL_FOOD")
        food_spans.append(span)
    
    # Process noun chunks
    for noun_chunk in doc.noun_chunks:
        if (noun_chunk.text.lower() not in NON_FOOD_WORDS and 
            not all(token.is_stop for token in noun_chunk)):
            # Check if chunk is part of another span
            if not any(span.start <= noun_chunk.start < span.end or 
                      span.start < noun_chunk.end <= span.end 
                      for span in food_spans + original_ents):
                food_spans.append(spacy.tokens.Span(doc, noun_chunk.start, noun_chunk.end, label="POTENTIAL_FOOD"))
    
    # Look for foods after quantities
    for ent in original_ents:
        if ent.label_ == "QUANTITY" and ent.end < len(doc):
            token = doc[ent.end]
            if token.pos_ == "NOUN" and token.ent_type_ == "":
                end = ent.end + 1
                while end < len(doc) and (doc[end].dep_ == "compound" or doc[end].pos_ == "NOUN"):
                    end += 1
                
                if token.text.lower() not in NON_FOOD_WORDS:
                    food_spans.append(spacy.tokens.Span(doc, ent.end, end, label="POTENTIAL_FOOD"))
    
    # Filter overlaps and set entities
    doc.ents = filter_spans(original_ents + food_spans)
    return doc

# Function to query USDA API
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

# Process a single food log entry
# Process a single food log entry
def process_food_text(text):
    doc = nlp(text)
    
    # Collect entities and their positions
    entities = {"food_items": [], "quantities": [], "meal_times": [], "meal_types": []}
    entity_positions = {}
    
    # Map to store time to meal type associations
    time_to_meal_type = {}
    
    for ent in doc.ents:
        if ent.label_ == "POTENTIAL_FOOD":
            entities["food_items"].append(ent.text)
            entity_positions[ent.text] = {"type": "food", "start": ent.start, "end": ent.end}
        elif ent.label_ == "QUANTITY":
            entities["quantities"].append(ent.text)
            entity_positions[ent.text] = {"type": "quantity", "start": ent.start, "end": ent.end}
        elif ent.label_ == "MEAL_TIME":
            # Check if it's a meal type or a time
            if re.search(r'\d', ent.text):  # Contains a digit, likely a time
                standardized_time = standardize_time(ent.text)
                entities["meal_times"].append(standardized_time)
                entity_positions[standardized_time] = {"type": "meal_time", "start": ent.start, "end": ent.end}
            else:  # No digit, likely a meal type (breakfast, lunch, dinner)
                entities["meal_types"].append(ent.text.lower())
                entity_positions[ent.text.lower()] = {"type": "meal_type", "start": ent.start, "end": ent.end}
    
    # Find associations between time and meal types based on proximity in text
    for time in entities["meal_times"]:
        time_pos = entity_positions[time]["start"]
        closest_type = None
        min_distance = float('inf')
        
        for meal_type in entities["meal_types"]:
            meal_type_pos = entity_positions[meal_type]["start"]
            distance = abs(time_pos - meal_type_pos)
            if distance < min_distance:
                min_distance = distance
                closest_type = meal_type
        
        # If we found a close meal type, associate them
        if closest_type and min_distance < 15:  # 15 tokens is an arbitrary threshold
            time_to_meal_type[time] = closest_type
    
    # Query USDA API for each food item
    food_details = {}
    for food in entities["food_items"]:
        details = query_usda_for_food(food)
        food_details[food] = details
    
    # Temporary storage for associating foods with meal times
    time_foods = {}
    for time in entities["meal_times"]:
        time_foods[time] = []
    
    if not entities["meal_times"]:
        time_foods["unspecified"] = []
    
    # Associate foods with meal times based on proximity
    for food in entities["food_items"]:
        # Find closest meal time
        food_pos = entity_positions[food]["start"]
        closest_time = None
        min_distance = float('inf')
        
        for time in entities["meal_times"]:
            time_pos = entity_positions[time]["start"]
            distance = abs(food_pos - time_pos)
            if distance < min_distance:
                min_distance = distance
                closest_time = time
        
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
        
        # Add to appropriate time group
        if closest_time:
            time_foods[closest_time].append(food_entry)
        else:
            if "unspecified" not in time_foods:
                time_foods["unspecified"] = []
            time_foods["unspecified"].append(food_entry)
    
    # Now reshape the data to the desired output format
    restructured_meals = {}
    
    # First, handle explicit meal types with their associated times
    for time, meal_type in time_to_meal_type.items():
        if meal_type not in restructured_meals:
            restructured_meals[meal_type] = {
                "time": time,
                "meals": time_foods.get(time, [])
            }
    
    # For times without explicit meal types, infer the meal type
    for time in entities["meal_times"]:
        if time not in time_to_meal_type.values():
            # Infer meal type based on time
            hour = int(time.split(":")[0])
            inferred_type = None
            
            if 5 <= hour < 11:
                inferred_type = "breakfast"
            elif 11 <= hour < 16:
                inferred_type = "lunch"
            elif 16 <= hour < 22:
                inferred_type = "dinner"
            else:
                inferred_type = "snack"
            
            # Check if this meal type already exists
            if inferred_type in restructured_meals:
                # If there are no foods in the existing entry, just update the time
                if not restructured_meals[inferred_type]["meals"]:
                    restructured_meals[inferred_type]["time"] = time
                # Otherwise, create a numbered version of the meal type
                else:
                    count = 1
                    new_type = f"{inferred_type}{count}"
                    while new_type in restructured_meals:
                        count += 1
                        new_type = f"{inferred_type}{count}"
                    
                    restructured_meals[new_type] = {
                        "time": time,
                        "meals": time_foods.get(time, [])
                    }
            else:
                restructured_meals[inferred_type] = {
                    "time": time,
                    "meals": time_foods.get(time, [])
                }
    
    # For meal types without times, use a default time
    for meal_type in entities["meal_types"]:
        if meal_type not in restructured_meals:
            # Assign a default time based on meal type
            default_time = ""
            if meal_type == "breakfast":
                default_time = "08:00"
            elif meal_type == "lunch":
                default_time = "12:00"
            elif meal_type == "dinner":
                default_time = "19:00"
            else:
                default_time = "15:00"  # Default for other meal types
            
            # Find foods associated with this meal type directly
            meal_type_foods = []
            meal_type_pos = entity_positions[meal_type]["start"]
            
            for food in entities["food_items"]:
                food_pos = entity_positions[food]["start"]
                distance = abs(food_pos - meal_type_pos)
                if distance < 10:  # Close proximity to the meal type
                    # Find associated quantities for this food
                    associated_quantities = [
                        qty for qty in entities["quantities"]
                        if 0 <= entity_positions[food]["start"] - entity_positions[qty]["end"] <= 3
                    ]
                    
                    # Create food entry
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
                    
                    meal_type_foods.append({
                        "item": food,
                        "quantity": associated_quantities if associated_quantities else ["1 serving"],
                        "details": serializable_details
                    })
            
            restructured_meals[meal_type] = {
                "time": default_time,
                "meals": meal_type_foods
            }
    
    # Handle the unspecified case
    if "unspecified" in time_foods and time_foods["unspecified"]:
        restructured_meals["other"] = {
            "time": "",
            "meals": time_foods["unspecified"]
        }
    
    return {"meals": restructured_meals}

# Calculate nutritional summary for the new structure
def calculate_nutritional_summary(meals):
    summary = {}
    
    for meal_type, meal_data in meals.items():
        for food in meal_data["meals"]:
            if "details" in food and "nutrients" in food["details"]:
                for nutrient in food["details"]["nutrients"]:
                    try:
                        nutrient_name = nutrient.get("name", "")
                        value = float(nutrient.get("value", 0))
                        unit = nutrient.get("unit", "")
                        
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
        for meal_type, meal_data in results["meals"].items():
            if meal_type in all_results["meals"]:
                # If meal type already exists, append foods to the existing entry
                all_results["meals"][meal_type]["meals"].extend(meal_data["meals"])
            else:
                # Otherwise, add the new meal type
                all_results["meals"][meal_type] = meal_data
    
    # Calculate nutritional summary
    all_results["nutritional_summary"] = calculate_nutritional_summary(all_results["meals"])
    
    return all_results

# Function to standardize time formats
def standardize_time(time_text):
    time_patterns = [
        r'(\d{1,2}):?(\d{2})?\s*(am|pm)?',
        r'(\d{1,2})\s*(am|pm)'
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, time_text, re.IGNORECASE)
        if match:
            groups = match.groups()
            hour = int(groups[0])
            
            # Handle minutes
            minutes = groups[1] if groups[1] else "00"
            minutes = int(minutes) if minutes else 0
            
            # Handle am/pm
            period = groups[2].lower() if groups[2] else ""
            if period == "pm" and hour < 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
                
            # Format time in 24-hour format
            return f"{hour:02d}:{minutes:02d}"
    
    # If no pattern matches, return the original text
    return time_text

# Setup NLP pipeline at module level
setup_nlp_pipeline()

if __name__ == "__main__":
    input_log = [
        "At 8:22 AM, I had 20 gram salmon and 2 slices of salad with roasted beef.",
        "For lunch around 12:31, I ate a chicken sandwich with tomato.",
        "My dinner at 7 PM served with 3 egg and 2 slices of bacon.",
    ]
    
    # Process logs and print results
    results = process_food_log(input_log)
    for i, log in enumerate(input_log, 1):
        print(f"{i}. {log}")
    
    print("\nJSON Output:")
    print(json.dumps(results, indent=2))