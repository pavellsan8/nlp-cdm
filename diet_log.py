import os
import spacy
import re
import json
import requests
from spacy.language import Language
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()
nlp = spacy.load("en_core_web_md")

class MealInfoExtractor:
    def __init__(self, nlp, name, api_key=None):
        self.time_pattern = re.compile(r'(\d{1,2}:\d{2}(?:\s*[AP]M)?|\d{1,2}\s*[AP]M)')
        self.quantity_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(gram|g|slice|slices|cup|cups|oz|ounce|ounces|egg|eggs|pieces?)')
        
        # Use sets for faster lookups
        self.meal_dict = {
            "breakfast": set(["breakfast", "morning", "am"]),
            "lunch": set(["lunch", "noon", "afternoon"]),
            "dinner": set(["dinner", "evening", "night", "pm"])
        }
        self.stop_words = set(["i", "me", "my", "myself", "and", "with", "a", "an", "the", "had", "ate", "of", "for", "in", "around", "at", "served"])
        
        # USDA API settings
        self.api_key = api_key
        self.usda_search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
        
        self.food_keywords = {
            "alfredo", "broccoli", "brownie", "cake", "carrot", "cereal", "cheese",
            "chicken", "chocolate", "coffee", "cookie", "corn", "couscous", "crab",
            "donut", "egg", "fajitas", "fries", "grilledcheese", "hotdog",
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
            "egg": "egg, whole, raw",
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
        
    def __call__(self, doc):
        doc._.meal_info = self.extract_meal_info(doc)
        return doc
    
    def extract_meal_info(self, doc):
        text = doc.text.lower() 
        
        time_match = self.time_pattern.search(text)
        meal_time = time_match.group(0) if time_match else None
        meal_type = self.determine_meal_type(text, meal_time)
        
        quantities = []
        quantity_spans = []
        
        quantity_matches = self.quantity_pattern.finditer(text)
        for match in quantity_matches:
            quantities.append(match.group(0))
            quantity_spans.append((match.start(), match.end()))
        
        food_items = self.extract_food_items(doc, quantity_spans)
        filtered_food_items = set()

        for food in food_items:
            cleaned_food = self.clean_food_text(food)
            if cleaned_food:
                filtered_food_items.add(cleaned_food)
        
        filtered_food_items = list(filtered_food_items)

        nutritional_info = {}
        if self.api_key and filtered_food_items:
            for food in filtered_food_items:
                nutritional_info[food] = self.get_nutritional_info(food)
        
        return {
            "meal_type": meal_type,
            "meal_time": meal_time,
            "quantities": quantities,
            "food_items": filtered_food_items,
            "nutritional_info": nutritional_info
        }
    
    def extract_food_items(self, doc, quantity_spans):
        food_items = []
        for ent in doc.ents:
            if any(start <= ent.start_char < end or start < ent.end_char <= end for start, end in quantity_spans):
                continue
            if self.is_likely_food(ent.text):
                food_items.append(ent.text)
        
        if not food_items:
            for chunk in doc.noun_chunks:
                if any(start <= chunk.start_char < end or start < chunk.end_char <= end for start, end in quantity_spans):
                    continue
                if self.is_likely_food(chunk.text):
                    food_items.append(chunk.text)
        return food_items
    
    def clean_food_text(self, text):
        words = text.lower().split()
        filtered_words = [word for word in words if word not in self.stop_words]
        
        if not filtered_words:
            return ""
        return " ".join(filtered_words)
    
    def determine_meal_type(self, text_lower, time_str):
        for meal, keywords in self.meal_dict.items():
            if any(keyword in text_lower for keyword in keywords):
                return meal
        if time_str:
            try:
                if ":" in time_str:
                    if "am" in time_str.lower() or "pm" in time_str.lower():
                        parts = time_str.lower().replace("am", "").replace("pm", "").strip().split(":")
                        hour = int(parts[0])
                        if "pm" in time_str.lower() and hour < 12:
                            hour += 12
                    else:
                        hour = int(time_str.split(":")[0])
                else:
                    parts = time_str.lower().replace("am", "").replace("pm", "").strip().split()
                    hour = int(parts[0])
                    if "pm" in time_str.lower() and hour < 12:
                        hour += 12 
                if 5 <= hour < 11:
                    return "breakfast"
                elif 11 <= hour < 15:
                    return "lunch"
                elif 17 <= hour < 22:
                    return "dinner"
                else:
                    return "snack"
            except:
                return "unknown"
        return "unknown"
    
    def is_likely_food(self, text):
        text_lower = text.lower()
        return any(word in self.food_keywords for word in text_lower.split())
    
    @lru_cache(maxsize=128)
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
        
    def process_logs(self, input_logs):
        grouped_results = {}
        
        for doc in self.nlp.pipe(input_logs):
            meal_info = doc._.meal_info
            meal_type = meal_info["meal_type"]
            
            if meal_type not in grouped_results:
                grouped_results[meal_type] = []
            
            meal_entry = {
                "log": doc.text,
                "meal_time": meal_info["meal_time"],
                "quantities": meal_info["quantities"],
                "food_items": meal_info["food_items"]
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
                            "nutrients": food_nutrients
                        })
                
                meal_entry["food_details"] = food_details
                meal_entry["total_nutrients"] = total_nutrients
            
            grouped_results[meal_type].append(meal_entry)
        return grouped_results
    
    def generate_daily_summary(self, grouped_results):
        daily_totals = {nutrient: 0 for nutrient in self.extractor.important_nutrients}
        meal_counts = {}
        
        for meal_type, meals in grouped_results.items():
            meal_counts[meal_type] = len(meals)
            for meal in meals:
                if "total_nutrients" in meal:
                    for nutrient, value in meal["total_nutrients"].items():
                        daily_totals[nutrient] += value
        return {
            "daily_totals": daily_totals,
            "meal_counts": meal_counts
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