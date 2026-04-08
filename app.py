from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os, json, hashlib, secrets
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS

load_dotenv()

# Initialize Firebase

google_creds = os.getenv("GOOGLE_CREDENTIALS")
if google_creds:
    cred = credentials.Certificate(json.loads(google_creds))
else:
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

# ─── Helper: Per-User Subcollection ─────────────────────────

def user_collection(user_id, collection_name):
    return db.collection('users').document(user_id).collection(collection_name)

# ─── User Summary Functions ──────────────────────────────────

def get_user_summary(user_id='user_1'):
    doc = db.collection('users').document(user_id).get()
    base = {
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
        "effective_interventions": [],
        "intervention_summary": {
            "most_effective": None,
            "least_effective": None,
            "total_tried": 0,
            "favourite_category": None
        }
    }
    if doc.exists:
        data = doc.to_dict()
        # Merge with defaults — ensure all expected keys exist
        for key in base:
            if key not in data:
                data[key] = base[key]
        return data
    return base

def update_user_summary(user_id='user_1'):
    # Get all logs from user subcollections
    emotion_logs = [d.to_dict() for d in user_collection(user_id, 'emotion_logs').stream()]
    sleep_logs = [d.to_dict() for d in user_collection(user_id, 'sleep_logs').stream()]
    diet_logs = [d.to_dict() for d in user_collection(user_id, 'diet_logs').stream()]

    summary = get_user_summary(user_id)

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

    # Update intervention patterns
    intervention_logs = [d.to_dict() for d in user_collection(user_id, 'intervention_logs').stream()]
    if intervention_logs:
        # Most effective (highest average rating)
        from collections import defaultdict
        ratings_by_id = defaultdict(list)
        categories_count = defaultdict(int)
        for log in intervention_logs:
            iid = log.get('intervention_id', log.get('intervention_name', ''))
            rating = log.get('rating', 0)
            if iid and rating:
                ratings_by_id[iid].append(rating)
            cat = log.get('intervention_type', '')
            if cat:
                categories_count[cat] += 1

        if ratings_by_id:
            avg_ratings = {k: sum(v)/len(v) for k, v in ratings_by_id.items()}
            best = max(avg_ratings, key=avg_ratings.get)
            worst = min(avg_ratings, key=avg_ratings.get)
            summary['intervention_summary'] = {
                'most_effective': best,
                'least_effective': worst,
                'total_tried': len(ratings_by_id),
                'favourite_category': max(categories_count, key=categories_count.get) if categories_count else None
            }

    summary['last_updated'] = datetime.now().strftime("%Y-%m-%d")

    # Save updated summary
    db.collection('users').document(user_id).set(summary)
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

    user_id = data.get('user_id', 'user_1')
    user_text = data.get('text', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    # Get user summary for personalization
    summary = get_user_summary(user_id)
    
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
    user_collection(user_id, 'emotion_logs').add({
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

    user_id = data.get('user_id', 'user_1')
    hours = data.get('hours', 0)
    quality = data.get('quality', 'unknown')
    notes = data.get('notes', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    summary = get_user_summary(user_id)

    # Get recent sleep logs for personalisation
    recent_sleep = [d.to_dict() for d in user_collection(user_id, 'sleep_logs').order_by('date', direction='DESCENDING').limit(5).stream()]
    phase_sleep = [l for l in recent_sleep if l.get('phase') == cycle_phase]
    phase_avg = round(sum(l.get('hours', 0) for l in phase_sleep) / len(phase_sleep), 1) if phase_sleep else None
    personalised = len(recent_sleep) >= 5

    prompt = f"""
    You are a compassionate women's health assistant.

    The user is in their {cycle_phase} phase (day {cycle_day}).
    They slept {hours} hours and rated their sleep quality as {quality}.
    {f'Additional notes: {notes}' if notes else ''}
    Their historical average sleep is {summary['sleep_patterns']['average_hours']} hours.
    {f'Their average sleep during {cycle_phase} phase is {phase_avg} hours.' if phase_avg else ''}

    Based on their cycle phase, sleep data, and history:
    1. Explain in 1-2 sentences why their sleep may be affected hormonally right now
    2. Give one specific actionable tip to improve their sleep tonight

    Respond in JSON only, no other text:
    {{
        "insight": "hormonal explanation here",
        "tip": "actionable tip here"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        result = json.loads(response.choices[0].message.content)
    except Exception:
        result = {
            "insight": "Your sleep is important during your " + cycle_phase + " phase.",
            "tip": "Try to keep a consistent bedtime tonight."
        }

    result['personalised'] = personalised

    # Save to Firebase
    user_collection(user_id, 'sleep_logs').add({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hours": hours,
        "quality": quality,
        "notes": notes,
        "phase": cycle_phase,
        "cycle_day": cycle_day
    })

    # Update summary
    update_user_summary(user_id)

    return jsonify(result)

@app.route('/log_diet', methods=['POST'])
def log_diet():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    user_id = data.get('user_id', 'user_1')
    cravings = data.get('cravings', [])
    craving = data.get('craving', '')
    # Support both old string format and new array format
    if not cravings and craving:
        cravings = [craving]
    craving_str = ', '.join(cravings) if cravings else craving
    ate = data.get('ate', '')
    body_feel = data.get('body_feel', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    summary = get_user_summary(user_id)

    # Get recent diet logs for personalisation
    recent_diet = [d.to_dict() for d in user_collection(user_id, 'diet_logs').order_by('date', direction='DESCENDING').limit(5).stream()]
    personalised = len(recent_diet) >= 5
    phase_cravings = [l.get('craving', '') for l in recent_diet if l.get('phase') == cycle_phase and l.get('craving')]

    prompt = f"""
    You are a compassionate women's health assistant.

    The user is in their {cycle_phase} phase (day {cycle_day}).
    They are craving: "{craving_str}"
    They ate: "{ate}"
    They feel: "{body_feel}"
    Their most common historical craving is: {summary['diet_patterns']['most_common_craving']}
    {f'Common cravings in their {cycle_phase} phase: {", ".join(phase_cravings[:3])}' if phase_cravings else ''}

    Based on their cycle phase, dietary input, and history:
    1. Explain in 1-2 sentences why they might be experiencing these cravings hormonally
    2. Give one specific food swap or mindful eating tip for this phase

    Respond in JSON only, no other text:
    {{
        "insight": "hormonal explanation here",
        "tip": "actionable tip here"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        result = json.loads(response.choices[0].message.content)
    except Exception:
        result = {
            "insight": "Cravings during your " + cycle_phase + " phase are completely normal and driven by hormones.",
            "tip": "Try pairing your craving with something nutrient-dense."
        }

    result['personalised'] = personalised

    # Save to Firebase
    user_collection(user_id, 'diet_logs').add({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "craving": craving_str,
        "cravings": cravings,
        "ate": ate,
        "body_feel": body_feel,
        "phase": cycle_phase,
        "cycle_day": cycle_day
    })

    # Update summary
    update_user_summary(user_id)

    return jsonify(result)

@app.route('/report_period', methods=['POST'])
def report_period():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    user_id = data.get('user_id', 'user_1')
    period_date = data.get('period_date')
    if not period_date:
        return jsonify({"error": "period_date is required"}), 400

    # Get existing period dates from Firebase
    user_doc = db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    period_dates = user_data.get('period_dates', [])

    if period_date not in period_dates:
        period_dates.append(period_date)
        period_dates.sort()

    avg_cycle = calculate_average_cycle(period_dates)

    db.collection('users').document(user_id).set({
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
    user_id = request.args.get('user_id', 'user_1')
    user_doc = db.collection('users').document(user_id).get()
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

    user_id = data.get('user_id', 'user_1')
    intervention_id = data.get('intervention_id', '')
    intervention_name = data.get('intervention_name', data.get('intervention', ''))
    intervention_type = data.get('intervention_type', '')
    rating = data.get('rating', 0)
    phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)
    duration_seconds = data.get('duration_seconds', 0)

    # Save full log to intervention_logs subcollection
    user_collection(user_id, 'intervention_logs').add({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'intervention_id': intervention_id,
        'intervention_name': intervention_name,
        'intervention_type': intervention_type,
        'rating': rating,
        'phase': phase,
        'cycle_day': cycle_day,
        'duration_seconds': duration_seconds
    })

    user_doc = db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}

    if rating >= 4:
        effective = user_data.get('effective_interventions', [])
        if intervention_name not in effective:
            effective.append(intervention_name)
        db.collection('users').document(user_id).set({
            'effective_interventions': effective
        }, merge=True)

    if rating <= 2:
        ineffective = user_data.get('ineffective_interventions', [])
        if intervention_name not in ineffective:
            ineffective.append(intervention_name)
        db.collection('users').document(user_id).set({
            'ineffective_interventions': ineffective
        }, merge=True)

    return jsonify({"status": "ok", "message": "Thanks for rating \u2014 Kira will remember this"})

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

    user_id = data.get('user_id', 'user_1')
    user_message = data.get('message', '')
    cycle_phase = data.get('phase', 'unknown')
    cycle_day = data.get('cycle_day', 0)

    # Get conversation history from Firebase
    history_ref = user_collection(user_id, 'chat_history').order_by(
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
    summary = get_user_summary(user_id)

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
    - You have access to the user's historical patterns. When relevant, reference
      that Kira has noticed patterns: e.g. 'I've noticed you tend to feel more
      anxious in your luteal phase — is that what's happening now?' Use the
      intervention library to make specific suggestions by name, e.g. 'Want to
      try the 5-4-3-2-1 grounding exercise?' or 'The urge surfing timer might
      help with that craving.'
    - Most effective intervention for this user: {summary.get('intervention_summary', {}).get('most_effective', 'unknown')}
    - Least effective intervention: {summary.get('intervention_summary', {}).get('least_effective', 'unknown')}
    - Total interventions tried: {summary.get('intervention_summary', {}).get('total_tried', 0)}
    - Favourite category: {summary.get('intervention_summary', {}).get('favourite_category', 'unknown')}
    - If the user needs a visual intervention, end your message with
      exactly one of these tags on a new line:
      [BREATHING] or [URGE_SURF] or [HAPTIC] or [GROUNDING] or [JOURNAL]
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
    for tag in ['[BREATHING]', '[URGE_SURF]', '[HAPTIC]', '[GROUNDING]', '[JOURNAL]']:
        if tag in ai_message:
            intervention_trigger = tag.strip('[]')
            clean_message = ai_message.replace(tag, '').strip()

    # Save user message to Firebase
    user_collection(user_id, 'chat_history').add({
        'role': 'user',
        'content': user_message,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    # Save AI response to Firebase
    user_collection(user_id, 'chat_history').add({
        'role': 'assistant',
        'content': clean_message,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

    # Save emotion log silently
    user_collection(user_id, 'emotion_logs').add({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "emotion": detected_emotion,
        "user_text": user_message,
        "phase": cycle_phase,
        "cycle_day": cycle_day,
        "intervention": ""
    })

    # Update user summary periodically
    update_user_summary(user_id)

    return jsonify({
        "message": clean_message,
        "emotion": detected_emotion,
        "intervention_trigger": intervention_trigger
    })

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}
    user_id = data.get('user_id', 'user_1')
    # Delete all chat history for this user
    docs = user_collection(user_id, 'chat_history').stream()
    for doc in docs:
        doc.reference.delete()
    return jsonify({"message": "Chat cleared"})

# ─── Authentication ─────────────────────────────────────────

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return salt, hashed

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    # Validate username
    if not username or len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if len(username) > 20:
        return jsonify({"error": "Username must be 20 characters or fewer"}), 400
    if not username.isalnum() and '_' not in username:
        return jsonify({"error": "Username can only contain letters, numbers, and underscores"}), 400

    # Validate password
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    # Check if username already taken
    existing = db.collection('accounts').document(username).get()
    if existing.exists:
        return jsonify({"error": "Username already taken"}), 409

    # Create account
    salt, hashed = hash_password(password)
    db.collection('accounts').document(username).set({
        'username': username,
        'password_hash': hashed,
        'salt': salt,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    # Create user profile document
    db.collection('users').document(username).set({
        'username': username,
        'onboarding_complete': False
    })

    return jsonify({"status": "ok", "username": username, "user_id": username})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Look up account
    account = db.collection('accounts').document(username).get()
    if not account.exists:
        return jsonify({"error": "Invalid username or password"}), 401

    account_data = account.to_dict()
    salt = account_data.get('salt', '')
    stored_hash = account_data.get('password_hash', '')

    # Verify password
    _, check_hash = hash_password(password, salt)
    if check_hash != stored_hash:
        return jsonify({"error": "Invalid username or password"}), 401

    # Check if onboarding is complete
    user_doc = db.collection('users').document(username).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    onboarding_complete = user_data.get('onboarding_complete', False)

    return jsonify({
        "status": "ok",
        "username": username,
        "user_id": username,
        "onboarding_complete": onboarding_complete
    })

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

    user_id = data.get('user_id', 'user_1')

    db.collection('users').document(user_id).set({
        'name': name,
        'age': age,
        'last_period_date': last_period_date,
        'average_cycle_length': cycle_length,
        'period_dates': [last_period_date],
        'current_moods': current_moods,
        'onboarding_complete': True
    }, merge=True)

    return jsonify({"status": "ok", "name": name})

# ─── Intervention Library ───────────────────────────────────

INTERVENTION_LIBRARY = {
    "breathing": [
        {"id": "box_breathing", "name": "Box Breathing", "description": "4-4-4-4 breath pattern to calm your nervous system", "duration": "4 min", "best_for": ["anxiety", "stress", "overwhelmed"], "icon": "\U0001f32c\ufe0f"}
    ],
    "urge_surfing": [
        {"id": "urge_surf", "name": "Urge Surfing", "description": "Ride the wave of a craving without acting on it", "duration": "5–15 min", "best_for": ["craving", "irritability"], "icon": "\U0001f30a"}
    ],
    "haptic": [
        {"id": "haptic_reset", "name": "Haptic Reset", "description": "Rhythmic vibration patterns to calm or energise your nervous system", "duration": "2 min", "best_for": ["anxiety", "stress", "low_motivation"], "icon": "\U0001f4f3"}
    ],
    "grounding": [
        {"id": "54321", "name": "5-4-3-2-1 Grounding", "description": "Use your senses to anchor yourself to the present moment", "duration": "3 min", "best_for": ["anxiety", "rumination", "overwhelmed"], "icon": "\U0001f33f"}
    ],
    "journaling": [
        {"id": "phase_reflect", "name": "Phase Reflection", "description": "Guided journaling prompts tailored to your current cycle phase", "duration": "5 min", "best_for": ["low_motivation", "rumination", "social_sensitivity"], "icon": "\U0001f4d3"}
    ]
}

@app.route('/intervention_library', methods=['GET'])
def intervention_library():
    user_id = request.args.get('user_id', 'user_1')
    # Get user's ratings from intervention_logs
    logs = [d.to_dict() for d in user_collection(user_id, 'intervention_logs').stream()]
    user_doc = db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}
    effective = user_data.get('effective_interventions', [])

    # Calculate average ratings per intervention
    from collections import defaultdict
    ratings_by_id = defaultdict(list)
    for log in logs:
        iid = log.get('intervention_id', '')
        rating = log.get('rating', 0)
        if iid and rating:
            ratings_by_id[iid].append(rating)

    avg_ratings = {k: round(sum(v)/len(v), 1) for k, v in ratings_by_id.items()}

    # Build response with user ratings attached
    library = {}
    for category, items in INTERVENTION_LIBRARY.items():
        enriched = []
        for item in items:
            entry = dict(item)
            entry['user_rating'] = avg_ratings.get(item['id'])
            entry['times_used'] = len(ratings_by_id.get(item['id'], []))
            entry['effective'] = item['name'] in effective
            enriched.append(entry)
        library[category] = enriched

    return jsonify(library)


@app.route('/personalised_intervention', methods=['GET'])
def personalised_intervention():
    user_id = request.args.get('user_id', 'user_1')
    emotion = request.args.get('emotion', '')
    phase = request.args.get('phase', '')

    # Fetch user's intervention logs
    logs = [d.to_dict() for d in user_collection(user_id, 'intervention_logs').stream()]

    # Find interventions rated >= 4 that match the emotion
    from collections import defaultdict
    good_for_emotion = defaultdict(list)
    for log in logs:
        rating = log.get('rating', 0)
        iid = log.get('intervention_id', '')
        if rating >= 4 and iid:
            good_for_emotion[iid].append(rating)

    personalised = []
    # Check which of the user's well-rated interventions match the emotion
    all_interventions = {}
    for category, items in INTERVENTION_LIBRARY.items():
        for item in items:
            all_interventions[item['id']] = {**item, 'category': category}

    for iid, ratings in good_for_emotion.items():
        if iid in all_interventions:
            info = all_interventions[iid]
            if emotion in info.get('best_for', []):
                personalised.append({
                    **info,
                    'avg_rating': round(sum(ratings)/len(ratings), 1),
                    'reason': "Based on what's helped you before"
                })

    # Sort by average rating descending
    personalised.sort(key=lambda x: x['avg_rating'], reverse=True)

    # If not enough personalised, fill with defaults
    if len(personalised) < 3:
        for iid, info in all_interventions.items():
            if emotion in info.get('best_for', []) and iid not in [p['id'] for p in personalised]:
                personalised.append({
                    **info,
                    'avg_rating': None,
                    'reason': "Popular for this phase"
                })
            if len(personalised) >= 3:
                break

    return jsonify({"recommendations": personalised[:3]})


@app.route('/home_data', methods=['GET'])
def home_data():
    user_id = request.args.get('user_id', 'user_1')
    user_doc = db.collection('users').document(user_id).get()
    if not user_doc.exists:
        return jsonify({"error": "No user data"}), 400

    user_data = user_doc.to_dict()
    last_period = user_data.get('last_period_date')
    avg_cycle = user_data.get('average_cycle_length', 28)

    if not last_period:
        return jsonify({"error": "No period data"}), 400

    cycle = calculate_cycle_phase(last_period, avg_cycle)
    phase = cycle['phase']

    # Default tendencies
    default_tendencies = {
        "menstrual": ["Low energy", "Introspective", "Rest needed", "High sensitivity"],
        "follicular": ["Rising energy", "Creative clarity", "Motivated", "Social openness"],
        "ovulatory": ["Peak confidence", "High energy", "Communicative", "Assertive"],
        "luteal": ["Higher sensitivity", "Craving intensity", "Lower energy", "Introspective"]
    }

    result = {**cycle, 'personalised': False, 'name': user_data.get('name', '')}

    # Check emotion logs for personalisation
    emotion_logs = [d.to_dict() for d in user_collection(user_id, 'emotion_logs').stream()]
    if len(emotion_logs) >= 14:
        phase_emotions = [l['emotion'] for l in emotion_logs if l.get('phase') == phase]
        if phase_emotions:
            from collections import Counter
            top_emotions = [e for e, _ in Counter(phase_emotions).most_common(3)]
            result['tendencies'] = top_emotions
            result['tendencies_note'] = "Based on your patterns"
            result['personalised'] = True
        else:
            result['tendencies'] = default_tendencies.get(phase, [])
            result['tendencies_note'] = "Typical for this phase"
    else:
        result['tendencies'] = default_tendencies.get(phase, [])
        result['tendencies_note'] = "Typical for this phase"

    # Check sleep logs
    sleep_logs = [d.to_dict() for d in user_collection(user_id, 'sleep_logs').stream()]
    phase_sleep = [l for l in sleep_logs if l.get('phase') == phase]
    if phase_sleep:
        avg_sleep = round(sum(l.get('hours', 0) for l in phase_sleep) / len(phase_sleep), 1)
        result['sleep'] = {'average_hours': avg_sleep, 'personalised': True}
    else:
        phase_averages = {"menstrual": 6.5, "follicular": 7.2, "ovulatory": 7.0, "luteal": 6.8}
        result['sleep'] = {'average_hours': phase_averages.get(phase, 7.0), 'personalised': False}

    # Check diet logs
    diet_logs = [d.to_dict() for d in user_collection(user_id, 'diet_logs').stream()]
    phase_diet = [l for l in diet_logs if l.get('phase') == phase]
    if phase_diet:
        cravings = [l.get('craving', '') for l in phase_diet if l.get('craving')]
        if cravings:
            from collections import Counter
            top_cravings = [c for c, _ in Counter(cravings).most_common(3)]
            result['diet'] = {'top_cravings': top_cravings, 'personalised': True}
        else:
            result['diet'] = {'top_cravings': [], 'personalised': False}
    else:
        result['diet'] = {'top_cravings': [], 'personalised': False}

    return jsonify(result)


@app.route('/save_journal', methods=['POST'])
def save_journal():
    data = request.get_json(force=True)
    if isinstance(data, list):
        data = {item[0]: item[1] for item in data}

    user_id = data.get('user_id', 'user_1')

    user_collection(user_id, 'journal_logs').add({
        'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'prompt': data.get('prompt', ''),
        'entry': data.get('entry', ''),
        'phase': data.get('phase', 'unknown'),
        'cycle_day': data.get('cycle_day', 0)
    })

    return jsonify({"status": "ok", "message": "Journal entry saved"})


@app.route('/recent_logs', methods=['GET'])
def recent_logs():
    user_id = request.args.get('user_id', 'user_1')
    log_type = request.args.get('type', 'sleep')
    limit_count = int(request.args.get('limit', 3))

    collection_map = {
        'sleep': 'sleep_logs',
        'diet': 'diet_logs',
        'emotion': 'emotion_logs',
        'journal': 'journal_logs',
        'intervention': 'intervention_logs'
    }

    coll_name = collection_map.get(log_type)
    if not coll_name:
        return jsonify({"error": "Invalid log type"}), 400

    try:
        docs = user_collection(user_id, coll_name).order_by('date', direction='DESCENDING').limit(limit_count).stream()
        logs = [d.to_dict() for d in docs]
    except Exception:
        # If no index exists, fall back to unordered
        docs = user_collection(user_id, coll_name).limit(limit_count).stream()
        logs = sorted([d.to_dict() for d in docs], key=lambda x: x.get('date', ''), reverse=True)

    return jsonify({"logs": logs})


@app.route('/insights', methods=['GET'])
def insights():
    user_id = request.args.get('user_id', 'user_1')

    result = {}

    # Emotion history
    emotion_logs = [d.to_dict() for d in user_collection(user_id, 'emotion_logs').stream()]
    result['emotion_history'] = sorted(emotion_logs, key=lambda x: x.get('date', ''), reverse=True)[:30]

    # Emotion breakdown
    if emotion_logs:
        from collections import Counter
        emotions = [l.get('emotion', '') for l in emotion_logs if l.get('emotion') and l['emotion'] != 'unknown']
        counts = Counter(emotions)
        total = sum(counts.values())
        result['emotion_breakdown'] = [{'emotion': e, 'count': c, 'percent': round(c/total*100)} for e, c in counts.most_common(6)]
    else:
        result['emotion_breakdown'] = []

    # Mood trend (from emotion logs that have mood_rating, or approximate from emotions)
    mood_logs = [l for l in emotion_logs if l.get('mood_rating')]
    result['mood_trend'] = sorted(mood_logs, key=lambda x: x.get('date', ''), reverse=True)[:7]

    # Sleep history
    sleep_logs = [d.to_dict() for d in user_collection(user_id, 'sleep_logs').stream()]
    result['sleep_history'] = sorted(sleep_logs, key=lambda x: x.get('date', ''), reverse=True)[:7]

    # Diet history
    diet_logs = [d.to_dict() for d in user_collection(user_id, 'diet_logs').stream()]
    result['diet_history'] = sorted(diet_logs, key=lambda x: x.get('date', ''), reverse=True)[:7]

    # Journal entries
    journal_logs = [d.to_dict() for d in user_collection(user_id, 'journal_logs').stream()]
    result['journal_entries'] = sorted(journal_logs, key=lambda x: x.get('date', ''), reverse=True)[:20]

    # Intervention stats
    intervention_logs = [d.to_dict() for d in user_collection(user_id, 'intervention_logs').stream()]
    if intervention_logs:
        from collections import defaultdict
        stats = defaultdict(lambda: {'ratings': [], 'count': 0})
        for log in intervention_logs:
            name = log.get('intervention_name', '')
            if name:
                stats[name]['ratings'].append(log.get('rating', 0))
                stats[name]['count'] += 1
        result['intervention_stats'] = [
            {'name': k, 'avg_rating': round(sum(v['ratings'])/len(v['ratings']), 1), 'times_used': v['count']}
            for k, v in stats.items()
        ]
        result['intervention_stats'].sort(key=lambda x: x['avg_rating'], reverse=True)
    else:
        result['intervention_stats'] = []

    return jsonify(result)


@app.route('/notification_check', methods=['GET'])
def notification_check():
    """App Inventor polls this to know what notifications to show.
    Returns a list of notification messages relevant right now."""
    user_id = request.args.get('user_id', 'user_1')
    hour = datetime.now().hour

    user_doc = db.collection('users').document(user_id).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}

    notifications = []

    # Morning check-in (7-9am)
    if 7 <= hour <= 9:
        notifications.append({
            "type": "morning",
            "title": "Good morning ✦",
            "message": "How are you feeling today? Take a moment to check in.",
            "action": "navigate:home"
        })

    # Sleep reminder (21-22 / 9-10pm)
    if 21 <= hour <= 22:
        notifications.append({
            "type": "sleep",
            "title": "Wind down 🌙",
            "message": "Ready to log your sleep? Your body will thank you.",
            "action": "navigate:sleep"
        })

    # Water reminder (12-13 / noon)
    if 12 <= hour <= 13:
        notifications.append({
            "type": "water",
            "title": "Stay hydrated 💧",
            "message": "Have you had enough water today? Check your intake.",
            "action": "navigate:diet"
        })

    # Period prediction (if period is coming in 1-2 days)
    last_period = user_data.get('last_period_date')
    avg_cycle = user_data.get('average_cycle_length', 28)
    if last_period:
        try:
            cycle_info = calculate_cycle_phase(last_period, avg_cycle)
            if cycle_info['next_period_in_days'] <= 2:
                notifications.append({
                    "type": "period",
                    "title": "Period approaching",
                    "message": "Your period is predicted in " + str(cycle_info['next_period_in_days']) + " day(s). Be gentle with yourself.",
                    "action": "navigate:cycle"
                })
        except Exception:
            pass

    return jsonify({"notifications": notifications})


if __name__ == '__main__':
    app.run(debug=True, port=5000)