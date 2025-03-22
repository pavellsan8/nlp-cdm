import os
import spacy
import re
import json
import requests
from spacy.language import Language
from dotenv import load_dotenv
from functools import lru_cache
from collections import defaultdict

load_dotenv()
nlp = spacy.load("en_core_web_md")

class MealInfoExtractor:
    def __init__(self, nlp, name, api_key=None):
        self.time_pattern = re.compile(r'(\d{1,2}:\d{2}(?:\s*[AP]M)?|\d{1,2}\s*[AP]M)')
        self.quantity_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(gram|g|slice|slices|cup|cups|oz|ounce|ounces|egg|eggs|pieces?)')
        
        # Use sets for faster lookups
        self.meal_dict = {
            "breakfast": {"breakfast", "morning", "am"},
            "lunch": {"lunch", "noon", "afternoon"},
            "dinner": {"dinner", "evening", "night", "pm"}
        }
        self.stop_words = {"i", "me", "my", "myself", "and", "with", "a", "an", "the", "had", "ate", "of", "for", "in", "around", "at", "served"}
        
        # USDA API settings
        self.api_key = api_key
        self.usda_search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        
        self.food_keywords = {
            "alfredo", "broccoli", "brownie", "cake", "carrot", "cereal", "cheese",
            "chicken", "chocolate", "coffee", "cookie", "corn", "couscous", "crab",
            "donut", "egg", "eggs", "fajitas", "fries", "grilledcheese", "hotdog",
            "icecream", "macncheese", "nachos", "nuggets", "rice", "salad",
            "salmon", "shrimp", "soup", "steak", "sushi", "tartare", "beef",
            "bacon", "tomato", "sandwich", "roasted"
        }
        
        self.food_mapping = {
            "alfredo": "alfredo sauce",
            "broccoli": "broccoli, raw",
            "brownie": "brownie, baked",
            "cake": "cake, baked",
            "carrot": "carrot, raw",
            "cereal": "cereal, dry",
            "cheese": "cheese, sliced",
            "chicken": "chicken breast, raw",
            "chocolate": "chocolate, dark",
            "coffee": "coffee, brewed",
            "cookie": "cookie, baked",
            "corn": "corn, raw",
            "couscous": "couscous, cooked",
            "crab": "crab, raw",
            "donut": "donut, fried",
            "egg": "egg, eggs, whole, raw",
            "fajitas": "fajitas, cooked",
            "fries": "fries, cooked",
            "grilledcheese": "grilled cheese sandwich",
            "hotdog": "hotdog, cooked",
            "icecream": "ice cream, frozen",
            "macncheese": "mac and cheese, cooked",
            "nachos": "nachos, baked",
            "nuggets": "chicken nuggets, fried",
            "rice": "rice, cooked",
            "salad": "salad, mixed greens",
            "salmon": "salmon, raw",
            "shrimp": "shrimp, raw",
            "soup": "soup, cooked",
            "steak": "steak, grilled",
            "sushi": "sushi, raw",
            "tartare": "tartare, raw",
            "beef": "beef, raw",
            "bacon": "bacon, raw",
            "tomato": "tomato, raw",
            "sandwich": "sandwich, prepared",
            "roasted": "roasted vegetables"
        }
        
        self.important_nutrients = [
            "Energy", "Protein", "Carbohydrates", "Fat",
            "Calcium, Ca", "Iron, Fe", "Magnesium, Mg",
            "Phosphorus, P", "Potassium, K", "Sodium, Na"
        ]
        
        self.important_nutrients_set = set(self.important_nutrients)
        
        self.standard_serving_sizes = {
            "gram": 100,  # 100g as standard
            "g": 100,
            "slice": 1,
            "slices": 1,
            "cup": 1,
            "cups": 1,
            "oz": 28.35,  # 1 oz = 28.35g
            "ounce": 28.35,
            "ounces": 28.35,
            "egg": 1,
            "eggs": 1,
            "piece": 1,
            "pieces": 1
        }
        
    def __call__(self, doc):
        doc._.meal_info = self.extract_meal_info(doc)
        return doc
    
    def extract_meal_info(self, doc):
        text = doc.text.lower()
        
        time_match = self.time_pattern.search(text)
        meal_time = time_match.group(0) if time_match else None
        meal_type = self.determine_meal_type(text, meal_time)
        
        quantity_matches = list(self.quantity_pattern.finditer(text))
        
        quantities = []
        quantity_spans = []
        quantity_objects = []
        
        for match in quantity_matches:
            qty_text = match.group(0)
            amount, unit = self.parse_quantity(qty_text)
            
            quantities.append(qty_text)
            span = (match.start(), match.end())
            quantity_spans.append(span)
            
            quantity_objects.append({
                "text": qty_text,
                "amount": amount,
                "unit": unit,
                "span": span
            })
        
        food_items, food_spans = self.extract_food_items_with_spans(doc, quantity_spans)
        filtered_food_items = []
        for i, food in enumerate(food_items):
            cleaned_food = self.clean_food_text(food)
            if cleaned_food:
                filtered_food_items.append({
                    "name": cleaned_food,
                    "span": food_spans[i]
                })
        
        food_quantity_pairs = self.match_quantities_to_foods(filtered_food_items, quantity_objects, text)
        
        nutritional_info = {}
        if self.api_key and food_quantity_pairs:
            for pair in food_quantity_pairs:
                food = pair["food"]
                quantity_data = pair["quantity"]
                
                base_nutrition = self.get_nutritional_info(food)
                if "error" not in base_nutrition:
                    scaled_nutrition = self.scale_nutritional_info(
                        base_nutrition, 
                        quantity_data["amount"], 
                        quantity_data["unit"]
                    )
                    nutritional_info[food] = scaled_nutrition
                else:
                    nutritional_info[food] = base_nutrition
        food_items_list = [item["name"] for item in filtered_food_items]
        
        return {
            "meal_type": meal_type,
            "meal_time": meal_time,
            "quantities": quantities,
            "food_items": food_items_list,
            "food_quantity_pairs": food_quantity_pairs,
            "nutritional_info": nutritional_info
        }
    
    def match_quantities_to_foods(self, food_items, quantities, text):
        if not food_items or not quantities:
            return []
            
        pairs = []
        
        sorted_foods = sorted(food_items, key=lambda x: x["span"][0])
        sorted_quantities = sorted(quantities, key=lambda x: x["span"][0])
        
        foods_to_match = sorted_foods.copy()
        
        for qty in sorted_quantities:
            if not foods_to_match:
                break
                
            qty_pos = (qty["span"][0] + qty["span"][1]) / 2
            closest_food = None
            min_distance = float('inf')
            
            for food in foods_to_match:
                food_pos = (food["span"][0] + food["span"][1]) / 2
                distance = abs(food_pos - qty_pos)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_food = food
            
            if closest_food and min_distance < 50:
                pairs.append({
                    "food": closest_food["name"],
                    "quantity": qty
                })
                foods_to_match.remove(closest_food)
        
        for food in foods_to_match:
            pairs.append({
                "food": food["name"],
                "quantity": {
                    "text": "1 serving",
                    "amount": 1,
                    "unit": "serving"
                }
            })
            
        return pairs
    
    def scale_nutritional_info(self, base_nutrition, amount, unit):
        scaled_nutrition = dict(base_nutrition)
        scaling_factor = 1.0
        
        if unit in self.standard_serving_sizes:
            standard_amount = self.standard_serving_sizes[unit]
            if unit in ["gram", "g", "oz", "ounce", "ounces"]:
                scaling_factor = amount / standard_amount
            else:
                scaling_factor = amount
        
        if "nutrients" in scaled_nutrition:
            nutrients = scaled_nutrition["nutrients"]
            scaled_nutrients = {}
            
            for nutrient_name, nutrient_data in nutrients.items():
                if "value" in nutrient_data:
                    scaled_data = dict(nutrient_data)
                    scaled_data["value"] = round(nutrient_data["value"] * scaling_factor, 2)
                    scaled_nutrients[nutrient_name] = scaled_data
            
            scaled_nutrition["nutrients"] = scaled_nutrients
        
        scaled_nutrition["scaling"] = {
            "original_amount": amount,
            "original_unit": unit,
            "scaling_factor": scaling_factor
        }
        
        return scaled_nutrition
    
    def extract_food_items_with_spans(self, doc, quantity_spans):
        food_items = []
        food_spans = []
        
        quantity_ranges = []
        for start, end in quantity_spans:
            quantity_ranges.append(range(start, end))
        
        def overlaps_quantity(start_char, end_char):
            char_range = range(start_char, end_char)
            return any(range(max(r.start, char_range.start), min(r.stop, char_range.stop)) 
                      for r in quantity_ranges)
        
        for ent in doc.ents:
            if not overlaps_quantity(ent.start_char, ent.end_char) and self.is_likely_food(ent.text):
                food_items.append(ent.text)
                food_spans.append((ent.start_char, ent.end_char))
        
        if not food_items:
            for chunk in doc.noun_chunks:
                if not overlaps_quantity(chunk.start_char, chunk.end_char) and self.is_likely_food(chunk.text):
                    food_items.append(chunk.text)
                    food_spans.append((chunk.start_char, chunk.end_char))
        
        return food_items, food_spans
    
    def clean_food_text(self, text):
        filtered_words = [word for word in text.lower().split() if word not in self.stop_words]
        return " ".join(filtered_words) if filtered_words else ""
    
    def determine_meal_type(self, text_lower, time_str):
        for meal, keywords in self.meal_dict.items():
            if any(keyword in text_lower for keyword in keywords):
                return meal
        
        if time_str:
            try:
                hour = None
                
                if ":" in time_str:
                    time_lower = time_str.lower()
                    if "am" in time_lower or "pm" in time_lower:
                        parts = time_lower.replace("am", "").replace("pm", "").strip().split(":")
                        hour = int(parts[0])
                        if "pm" in time_lower and hour < 12:
                            hour += 12
                    else:
                        hour = int(time_str.split(":")[0])
                else:
                    time_lower = time_str.lower()
                    parts = time_lower.replace("am", "").replace("pm", "").strip().split()
                    hour = int(parts[0])
                    if "pm" in time_lower and hour < 12:
                        hour += 12 
                
                if hour is not None:
                    if 5 <= hour < 11:
                        return "breakfast"
                    elif 11 <= hour < 15:
                        return "lunch"
                    elif 17 <= hour < 22:
                        return "dinner"
                    else:
                        return "snack"
            except:
                pass
        
        return "unknown"
    
    def is_likely_food(self, text):
        text_words = set(text.lower().split())
        return bool(text_words & self.food_keywords)
    
    @lru_cache(maxsize=256)  # Increased cache size
    def get_nutritional_info(self, food_item):
        if not self.api_key:
            return {"error": "API key not provided"}
    
        search_term = self.food_mapping.get(food_item.lower().replace(" ", ""), food_item)
        
        try:
            params = {
                "api_key": self.api_key,
                "query": search_term,
                "dataType": ["Survey (FNDDS)"],
                "pageSize": 1
            }
            
            response = requests.get(self.usda_search_url, params=params)
            if response.status_code != 200:
                return {"error": f"API request failed with status code {response.status_code}"}
            
            search_results = response.json()
            if not search_results.get("foods", []):
                return {"error": f"No results found for {food_item}"}
            
            food_data = search_results["foods"][0]
            food_id = food_data["fdcId"]
            
            nutrients = {}
            for nutrient in food_data.get("foodNutrients", []):
                if "nutrientName" in nutrient and "value" in nutrient:
                    name = nutrient["nutrientName"]
                    if name in self.important_nutrients_set:
                        nutrients[name] = {
                            "value": nutrient["value"],
                            "unit": nutrient.get("unitName", "")
                        }
            
            return {
                "food_name": food_data["description"],
                "fdcId": food_id,
                "source": food_data.get("dataType", ""),
                "nutrients": nutrients
            }
        except Exception as e:
            return {"error": str(e)}
    
    def parse_quantity(self, quantity_str):
        match = re.match(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', quantity_str)
        if match:
            amount = float(match.group(1))
            unit = match.group(2)
            return amount, unit
        return None, None

@Language.factory("meal_info_extractor")
def create_meal_info_extractor(nlp, name, api_key=None):
    return MealInfoExtractor(nlp, name, api_key)

class MealTracker:
    def __init__(self, nlp, api_key=None):
        self.nlp = nlp
        
        if not spacy.tokens.Doc.has_extension("meal_info"):
            spacy.tokens.Doc.set_extension("meal_info", default=None, force=True)
        
        if "meal_info_extractor" not in self.nlp.pipe_names:
            self.nlp.add_pipe("meal_info_extractor", last=True, config={"api_key": api_key})
        
        self.extractor = self.nlp.get_pipe("meal_info_extractor")
        
        self.recommended_nutrients = {
            "Protein": 50,  # grams
            "Carbohydrates": 300,  # grams
            "Fat": 70,  # grams
            "Calcium, Ca": 1000,  # mg
            "Iron, Fe": 18,  # mg
            "Magnesium, Mg": 400,  # mg
            "Phosphorus, P": 700,  # mg
            "Potassium, K": 4700,  # mg
            "Sodium, Na": 2300,  # mg
        }
    
    def process_logs(self, input_logs):
        grouped_results = defaultdict(list)
        for doc in self.nlp.pipe(input_logs):
            meal_info = doc._.meal_info
            meal_type = meal_info["meal_type"]
            
            meal_entry = {
                "log": doc.text,
                "meal_time": meal_info["meal_time"],
                "quantities": meal_info["quantities"],
                "food_items": meal_info["food_items"],
                "food_quantity_pairs": meal_info.get("food_quantity_pairs", [])
            }
            
            if meal_info.get("nutritional_info"):
                total_nutrients = {nutrient: 0 for nutrient in self.extractor.important_nutrients}
                food_details = []
                
                for food, nutrient_data in meal_info["nutritional_info"].items():
                    if "nutrients" in nutrient_data:
                        nutrients = nutrient_data["nutrients"]
                        
                        food_nutrients = {}
                        for nutrient in self.extractor.important_nutrients:
                            if nutrient in nutrients:
                                value = nutrients[nutrient].get("value", 0)
                                food_nutrients[nutrient] = value
                                total_nutrients[nutrient] += value
                        
                        food_details.append({
                            "food_name": nutrient_data.get("food_name", food),
                            "nutrients": food_nutrients,
                            "scaling": nutrient_data.get("scaling", {})
                        })
                
                meal_entry["food_details"] = food_details
                meal_entry["total_nutrients"] = total_nutrients
            
            grouped_results[meal_type].append(meal_entry)
        return dict(grouped_results)
    
    def generate_daily_summary(self, grouped_results):
        daily_totals = {nutrient: 0 for nutrient in self.extractor.important_nutrients}
        meal_counts = {}
        recommendations = {}

        for meal_type, meals in grouped_results.items():
            meal_counts[meal_type] = len(meals)
            for meal in meals:
                if "total_nutrients" in meal:
                    for nutrient, value in meal["total_nutrients"].items():
                        daily_totals[nutrient] += value
        
        for nutrient, total in daily_totals.items():
            if nutrient in self.recommended_nutrients:
                recommended_value = self.recommended_nutrients[nutrient]
                percentage = (total / recommended_value) * 100
                
                if percentage < 80:
                    recommendations[nutrient] = f"Your {nutrient} intake is {percentage:.2f}% of the recommended daily intake. Consider adding more foods rich in {nutrient.lower()}."
                elif percentage > 120:
                    recommendations[nutrient] = f"Your {nutrient} intake is {percentage:.2f}% higher than the recommended daily intake. Consider reducing high-{nutrient.lower()} foods."
                else:
                    recommendations[nutrient] = f"Your {nutrient} intake is within the recommended range."

        return {
            "daily_totals": daily_totals,
            "meal_counts": meal_counts,
            "recommendations": recommendations
        }

if __name__ == "__main__":
    USDA_API_KEY = os.getenv("API_KEY")
    tracker = MealTracker(nlp, USDA_API_KEY)
    
    input_logs = [
        "At 8:22 AM, I had 20 gram salmon and 2 slices of salad with roasted beef.",
        "For lunch around 12:31, I ate a chicken sandwich with tomato.",
        "My dinner at 7 PM served with 3 egg and 2 slices of bacon."
    ]
    
    grouped_results = tracker.process_logs(input_logs)
    daily_summary = tracker.generate_daily_summary(grouped_results)
    response = {
        "meals": grouped_results,
        "daily_summary": daily_summary
    }
    
    print(json.dumps(response, indent=4))