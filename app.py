from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os, json
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS

load_dotenv()

# Initialize Firebase
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

app = Flask(__name__)
CORS(app)

# ─── User Summary Functions ──────────────────────────────────

def get_user_summary():
    doc = db.collection('users').document('user_1').get()
    if doc.exists:
        return doc.to_dict()
    return {
        "last_updated": None,
        "cycle_summary": {
            "average_cycle_length": 28,
            "cycles_tracked": 0
        },
        "emotion_patterns": {
            "most_common_emotion": None,
            "highest_risk_phase": None,
            "highest_risk_days": []
        },
        "sleep_patterns": {
            "average_hours": None,
            "worst_phase": None,
            "average_quality": None
        },
        "diet_patterns": {
            "most_common_craving": None,
            "craving_phase": None
        },
        "effective_interventions": []
    }

def update_user_summary():
    # Get all logs
    emotion_logs = [d.to_dict() for d in db.collection('emotion_logs').stream()]
    sleep_logs = [d.to_dict() for d in db.collection('sleep_logs').stream()]
    diet_logs = [d.to_dict() for d in db.collection('diet_logs').stream()]

    summary = get_user_summary()

    # Update emotion patterns
    if emotion_logs:
        emotions = [l['emotion'] for l in emotion_logs]
        summary['emotion_patterns']['most_common_emotion'] = max(set(emotions), key=emotions.count)
        
        luteal_logs = [l for l in emotion_logs if l['phase'] == 'luteal']
        if luteal_logs:
            summary['emotion_patterns']['highest_risk_phase'] = 'luteal'
            days = [l['cycle_day'] for l in luteal_logs]
            summary['emotion_patterns']['highest_risk_days'] = list(set(days))

    # Update sleep patterns
    if sleep_logs:
        hours = [l['hours'] for l in sleep_logs]
        summary['sleep_patterns']['average_hours'] = round(sum(hours) / len(hours), 1)
        qualities = [l['quality'] for l in sleep_logs]
        summary['sleep_patterns']['average_quality'] = max(set(qualities), key=qualities.count)

    # Update diet patterns
    if diet_logs:
        cravings = [l['craving'] for l in diet_logs if l.get('craving')]
        if cravings:
            summary['diet_patterns']['most_common_craving'] = max(set(cravings), key=cravings.count)

    summary['last_updated'] = datetime.now().strftime("%Y-%m-%d")

    # Save updated summary
    db.collection('users').document('user_1').set(summary)
    return summary

# ─── Cycle Functions ─────────────────────────────────────────

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

def calculate_average_cycle(period_dates):
    if len(period_dates) < 2:
        return 28
    dates = sorted([datetime.strptime(d, "%Y-%m-%d") for d in period_dates])
    gaps = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
    return round(sum(gaps) / len(gaps))

# ─── Routes ──────────────────────────────────────────────────

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    user_text = data.get('text', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    # Get user summary for personalization
    summary = get_user_summary()
    
    summary_text = f"""
    User's historical patterns:
    - Most common emotion: {summary['emotion_patterns']['most_common_emotion']}
    - Highest risk phase: {summary['emotion_patterns']['highest_risk_phase']}
    - Highest risk days: {summary['emotion_patterns']['highest_risk_days']}
    - Average sleep: {summary['sleep_patterns']['average_hours']} hours
    - Most common craving: {summary['diet_patterns']['most_common_craving']}
    - Interventions that helped before: {summary['effective_interventions']}
    """

    prompt = f"""
    You are a compassionate mental wellness assistant for a women's health app.
    
    The user is currently in their {cycle_phase} phase (day {cycle_day} of their cycle).
    They have shared: "{user_text}"
    
    {summary_text}
    
    Use their historical patterns to personalize your response.
    If you notice they are in their typical high risk window, acknowledge it warmly.
    If you know which interventions helped them before, prioritize those.
    
    Do two things:
    1. Classify their primary emotional state as ONE of: anxiety, rumination,
       irritability, craving, low_motivation, shame, social_sensitivity
    2. Give one short, specific micro-intervention (2-3 sentences max)
       appropriate for their emotional state, cycle phase, and personal history.
    
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

    # Save log to Firebase
    db.collection('emotion_logs').add({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "emotion": result['emotion'],
        "user_text": user_text,
        "phase": cycle_phase,
        "cycle_day": cycle_day,
        "intervention": result['intervention']
    })

    return jsonify(result)

@app.route('/log_sleep', methods=['POST'])
def log_sleep():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    hours = data.get('hours', 0)
    quality = data.get('quality', 'unknown')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    summary = get_user_summary()

    prompt = f"""
    You are a compassionate women's health assistant.
    
    The user is in their {cycle_phase} phase (day {cycle_day}).
    They slept {hours} hours and rated their sleep quality as {quality}.
    Their historical average sleep is {summary['sleep_patterns']['average_hours']} hours.
    
    Based on their cycle phase, sleep data, and history:
    1. Explain in 1 sentence why their sleep may be affected hormonally right now
    2. Give one specific actionable tip to improve their sleep tonight
    
    Respond in JSON only, no other text:
    {{
        "insight": "hormonal explanation here",
        "tip": "actionable tip here"
    }}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    result = json.loads(response.choices[0].message.content)

    # Save to Firebase
    db.collection('sleep_logs').add({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hours": hours,
        "quality": quality,
        "phase": cycle_phase,
        "cycle_day": cycle_day
    })

    # Update summary
    update_user_summary()

    return jsonify(result)

@app.route('/log_diet', methods=['POST'])
def log_diet():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    craving = data.get('craving', '')
    ate = data.get('ate', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    summary = get_user_summary()

    prompt = f"""
    You are a compassionate women's health assistant.
    
    The user is in their {cycle_phase} phase (day {cycle_day}).
    They are craving: "{craving}"
    They ate: "{ate}"
    Their most common historical craving is: {summary['diet_patterns']['most_common_craving']}
    
    Based on their cycle phase, dietary input, and history:
    1. Explain in 1 sentence why they might be experiencing these cravings hormonally
    2. Give one specific food swap or mindful eating tip for this phase
    
    Respond in JSON only, no other text:
    {{
        "insight": "hormonal explanation here",
        "tip": "actionable tip here"
    }}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    result = json.loads(response.choices[0].message.content)

    # Save to Firebase
    db.collection('diet_logs').add({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "craving": craving,
        "ate": ate,
        "phase": cycle_phase,
        "cycle_day": cycle_day
    })

    # Update summary
    update_user_summary()

    return jsonify(result)

@app.route('/report_period', methods=['POST'])
def report_period():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    period_date = data.get('period_date')
    if not period_date:
        return jsonify({"error": "period_date is required"}), 400

    # Get existing period dates from Firebase
    user_doc = db.collection('users').document('user_1').get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    period_dates = user_data.get('period_dates', [])

    if period_date not in period_dates:
        period_dates.append(period_date)
        period_dates.sort()

    avg_cycle = calculate_average_cycle(period_dates)

    db.collection('users').document('user_1').set({
        'period_dates': period_dates,
        'average_cycle_length': avg_cycle,
        'last_period_date': max(period_dates)
    }, merge=True)

    return jsonify({
        "message": "Period reported successfully",
        "periods_tracked": len(period_dates),
        "average_cycle_length": avg_cycle,
        "last_period_date": max(period_dates)
    })

@app.route('/cycle_status', methods=['GET'])
def cycle_status():
    user_doc = db.collection('users').document('user_1').get()
    if not user_doc.exists:
        return jsonify({"error": "No period data recorded yet"}), 400

    user_data = user_doc.to_dict()
    last_period = user_data.get('last_period_date')
    avg_cycle = user_data.get('average_cycle_length', 28)

    if not last_period:
        return jsonify({"error": "No period data recorded yet"}), 400

    result = calculate_cycle_phase(last_period, avg_cycle)
    result['periods_tracked'] = len(user_data.get('period_dates', []))
    return jsonify(result)

@app.route('/rate_intervention', methods=['POST'])
def rate_intervention():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    intervention = data.get('intervention', '')
    rating = data.get('rating', 0)  # 1-5

    # If highly rated, add to effective interventions
    if rating >= 4:
        user_doc = db.collection('users').document('user_1').get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        effective = user_data.get('effective_interventions', [])
        if intervention not in effective:
            effective.append(intervention)
        db.collection('users').document('user_1').set({
            'effective_interventions': effective
        }, merge=True)

    return jsonify({"message": "Rating saved", "rating": rating})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running"})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)

    # Handle all formats App Inventor might send
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}
    elif isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        data = {}

    user_message = data.get('message', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    # Get conversation history from Firebase
    history_ref = db.collection('chat_history').order_by(
        'timestamp').limit_to_last(10)
    history_docs = history_ref.get()
    conversation_history = []
    for doc in history_docs:
        d = doc.to_dict()
        conversation_history.append({
            "role": d['role'],
            "content": d['content']
        })

    # Get user summary for personalization
    summary = get_user_summary()

    # System prompt — the AI's personality
    system_prompt = f"""    
    You are Kira, a warm and supportive companion for women's health and wellness.
    
    Your personality:
    - Talk like a caring, knowledgeable best friend — warm, casual, never clinical
    - Never say things like "I've detected that you are feeling anxious"
    - Instead naturally reflect emotions back like a friend would: 
      "that sounds really overwhelming" or "ugh that's so frustrating"
    - You have deep knowledge about how hormones affect mood, sleep, cravings 
      and energy — but you share this naturally in conversation, not as a lecture

    How to guide the conversation:
    - ALWAYS ask a follow up question first before suggesting anything
    - Be genuinely curious about the user's day and what happened
    - Let the user vent and feel heard before offering any help
    - Try and suggest an intervention after at least 2-3 exchanges
    - If you detect extreme emotions after the first message, offer an intervention.
    - If the user shares something vague like "I feel snappy", ask what happened
    - If the user shares something specific, reflect it back and ask how they're 
      feeling about it
    - Examples of good follow up questions:
      "what happened today that brought this on?"
      "how long have you been feeling like this?"
      "is there something specific that set it off or did it just creep up?"
      "how are you feeling in your body right now?"
    - Unless extreme emotions detected, only suggest a breathing exercise,
      urge surfing, or haptic reset after the user has shared enough 
      and it feels natural — like a friend saying 
      "do you want to try something that might help?"

    Current context:
    - The user is in their {cycle_phase} phase, day {cycle_day} of their cycle
    - Their most common emotion: {summary['emotion_patterns']['most_common_emotion']}
    - Their highest risk phase: {summary['emotion_patterns']['highest_risk_phase']}
    - Interventions that helped them before: {summary['effective_interventions']}
    
    Important rules:
    - Keep responses SHORT — 2-4 sentences max, like a real text conversation
    - Never use bullet points or lists
    - Never sound like a doctor or therapist
    - Don't suggest an intervention until at least the 3rd message exchange unless needed
    - If the user needs a visual intervention, end your message with 
      exactly one of these tags on a new line:
      [BREATHING] or [URGE_SURF] or [HAPTIC]
    - Only suggest an intervention if it feels completely natural
    """

    # Build messages array with full history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_message})

    # Add emotion detection instruction to the last user message
    messages[-1]["content"] = user_message + """

    (After your response, on a completely new line, write exactly: 
    EMOTION:one_word
    where one_word is the detected emotion from: anxiety, rumination, 
    irritability, craving, low_motivation, shame, social_sensitivity, positive
    This line is hidden from the user.)"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        max_tokens=250
    )

    full_response = response.choices[0].message.content

    # Split emotion from message
    detected_emotion = "unknown"
    ai_message = full_response

    if "EMOTION:" in full_response:
        parts = full_response.split("EMOTION:")
        ai_message = parts[0].strip()
        detected_emotion = parts[1].strip().lower().split()[0]

    # Check if AI suggested an intervention
    intervention_trigger = None
    clean_message = ai_message
    for tag in ['[BREATHING]', '[URGE_SURF]', '[HAPTIC]']:
        if tag in ai_message:
            intervention_trigger = tag.strip('[]')
            clean_message = ai_message.replace(tag, '').strip()

    # Save user message to Firebase
    db.collection('chat_history').add({
        'role': 'user',
        'content': user_message,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    # Save AI response to Firebase
    db.collection('chat_history').add({
        'role': 'assistant',
        'content': clean_message,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    # Save emotion log silently
    db.collection('emotion_logs').add({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "emotion": detected_emotion,
        "user_text": user_message,
        "phase": cycle_phase,
        "cycle_day": cycle_day,
        "intervention": ""
    })

    # Update user summary periodically
    update_user_summary()

    return jsonify({
        "message": clean_message,
        "emotion": detected_emotion,
        "intervention_trigger": intervention_trigger
    })

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    # Delete all chat history
    docs = db.collection('chat_history').stream()
    for doc in docs:
        doc.reference.delete()
    return jsonify({"message": "Chat cleared"})

@app.route('/save_onboarding', methods=['POST'])
def save_onboarding():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    name = data.get('name', '')
    age = data.get('age', 0)
    last_period_date = data.get('last_period_date', '')
    cycle_length = data.get('cycle_length', 28)
    current_moods = data.get('current_moods', [])

    db.collection('users').document('user_1').set({
        'name': name,
        'age': age,
        'last_period_date': last_period_date,
        'average_cycle_length': cycle_length,
        'period_dates': [last_period_date],
        'current_moods': current_moods,
        'onboarding_complete': True
    }, merge=True)

    return jsonify({"status": "ok", "name": name})

if __name__ == '__main__':
    app.run(debug=True, port=5000)