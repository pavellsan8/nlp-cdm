import os
import spacy
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from diet_log import MealTracker

load_dotenv()
nlp = spacy.load("en_core_web_md")

app = Flask(__name__)
USDA_API_KEY = os.getenv("API_KEY")
tracker = MealTracker(nlp, USDA_API_KEY)

@app.route('/process_diet_logs', methods=['POST'])
def process_logs():
    try:
        data = request.get_json()
        if not data or 'logs' not in data:
            return jsonify({"error": "Invalid input. Please provide 'logs' key."}), 400
        
        input_logs = data['logs']
        grouped_results = tracker.process_logs(input_logs)
        daily_summary = tracker.generate_daily_summary(grouped_results)
        
        response = {
            "meals": grouped_results,
            "daily_summary": daily_summary
        }
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)