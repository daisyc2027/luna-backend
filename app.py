from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os, json
from pathlib import Path

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

app = Flask(__name__)

CYCLE_DATA_FILE = "cycle_data.json"

# ─── Helper Functions ───────────────────────────────────────

def load_cycle_data():
    if Path(CYCLE_DATA_FILE).exists():
        with open(CYCLE_DATA_FILE, "r") as f:
            return json.load(f)
    return {"period_dates": [], "average_cycle_length": 28}

def save_cycle_data(data):
    with open(CYCLE_DATA_FILE, "w") as f:
        json.dump(data, f)

def calculate_average_cycle(period_dates):
    if len(period_dates) < 2:
        return 28
    dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in period_dates])
    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    return round(sum(gaps) / len(gaps))

def calculate_cycle_phase(last_period_date_str, cycle_length=28):
    last_period = datetime.strptime(last_period_date_str, "%Y-%m-%d")
    today = datetime.today()
    days_since_period = (today - last_period).days
    cycle_day = (days_since_period % cycle_length) + 1
    if cycle_day <= 5:
        phase = "menstrual"
    elif cycle_day <= 13:
        phase = "follicular"
    elif cycle_day <= 16:
        phase = "ovulatory"
    else:
        phase = "luteal"
    return {
        "cycle_day": cycle_day,
        "phase": phase,
        "cycle_length": cycle_length,
        "next_period_in_days": cycle_length - cycle_day + 1
    }

# ─── Routes ─────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    
    data = request.get_json(force=True)
    if isinstance(data, list):
            # App Inventor sends as list of [key, value] pairs
         data = {item[0]: item[1] for item in data}

    user_text = data.get('text', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    prompt = f"""
    You are a compassionate mental wellness assistant for a women's health app.
    
    The user is currently in their {cycle_phase} phase (day {cycle_day} of their cycle).
    They have shared: "{user_text}"
    
    Do two things:
    1. Classify their primary emotional state as ONE of: anxiety, rumination,
       irritability, craving, low_motivation, shame, social_sensitivity
    2. Give them one short, specific micro-intervention (2-3 sentences max)
       appropriate for BOTH their emotional state AND their cycle phase.
    
    Respond in JSON only, no other text:
    {{
        "emotion": "emotion_label",
        "intervention": "your intervention text here",
        "confidence": 0.0
    }}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    result = json.loads(response.choices[0].message.content)
    return jsonify(result)

@app.route('/cycle', methods=['POST'])
def cycle():
    data = request.get_json()
    last_period = data.get('last_period_date')
    cycle_length = data.get('cycle_length', 28)
    if not last_period:
        return jsonify({"error": "last_period_date is required (YYYY-MM-DD format)"}), 400
    result = calculate_cycle_phase(last_period, cycle_length)
    return jsonify(result)

@app.route('/report_period', methods=['POST'])
def report_period():
    data = request.get_json()
    period_date = data.get('period_date')
    if not period_date:
        return jsonify({"error": "period_date is required"}), 400
    cycle_data = load_cycle_data()
    if period_date not in cycle_data["period_dates"]:
        cycle_data["period_dates"].append(period_date)
        cycle_data["period_dates"].sort()
    cycle_data["average_cycle_length"] = calculate_average_cycle(cycle_data["period_dates"])
    cycle_data["last_period_date"] = max(cycle_data["period_dates"])
    save_cycle_data(cycle_data)
    return jsonify({
        "message": "Period reported successfully",
        "periods_tracked": len(cycle_data["period_dates"]),
        "average_cycle_length": cycle_data["average_cycle_length"],
        "last_period_date": cycle_data["last_period_date"]
    })

@app.route('/cycle_status', methods=['GET'])
def cycle_status():
    cycle_data = load_cycle_data()
    if not cycle_data.get("last_period_date"):
        return jsonify({"error": "No period data recorded yet"}), 400
    result = calculate_cycle_phase(
        cycle_data["last_period_date"],
        cycle_data["average_cycle_length"]
    )
    result["periods_tracked"] = len(cycle_data["period_dates"])
    result["average_cycle_length"] = cycle_data["average_cycle_length"]
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)