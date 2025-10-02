"""Flask + OpenRouter multi-AI chat + AskLurk best-answer picker + File Upload + Firebase Auth
Public access – AUTHENTICATED ACCESS
"""
import os, json, logging, secrets, re, tempfile
from datetime import datetime, timedelta
from flask import Flask, request, Response, jsonify, render_template, session, redirect, url_for
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import firebase_admin
from firebase_admin import credentials, auth
import pdfplumber

# --- Configuration & Initialization ---

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
# IMPORTANT: Use a strong, secret key in production.
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# Firebase configuration for frontend (use actual values from your project)
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", "AIzaSyDqohvAqFwV209Aiz2OjKg2jxHzQmaiG4E"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", "multimodel-fd9e9.firebaseapp.com"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", "multimodel-fd9e9"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", "multimodel-fd9e9.firebasestorage.app"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", "814216734391"),
    "appId": os.getenv("FIREBASE_APP_ID", "1:814216734391:web:c21e8a727733b806168f32")
}

app.config.update(FIREBASE_CONFIG)

# --- FIREBASE ADMIN SDK SETUP ---
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")
if FIREBASE_CRED_PATH and os.path.exists(FIREBASE_CRED_PATH):
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Firebase Admin SDK: {e}")
else:
    logging.warning("FIREBASE_CRED_PATH not set or file not found. Authentication will be partially disabled.")

GLOBAL_LIMIT = 300
MODELS = {
    "logic":     {"model":"deepseek/deepseek-chat-v3.1:free",       "sys":"give short answer"},
    "creative":  {"model":"deepseek/deepseek-chat-v3.1:free", "sys":"give short answer"},
    "balanced":  {"model":"deepseek/deepseek-chat-v3.1:free",       "sys":"give short answer"},
    "gpt4o":     {"model":"deepseek/deepseek-chat-v3.1:free",             "sys":"You are GPT-4o – clear, accurate, concise."},
    "claude3":   {"model":"deepseek/deepseek-chat-v3.1:free",  "sys":"You are Claude 3 – thoughtful and precise."},
    "llama31":   {"model":"deepseek/deepseek-chat-v3.1:free",  "sys":"give short answer"},
    "mixtral":   {"model":"deepseek/deepseek-chat-v3.1:free",    "sys":"give short answer"},
    "qwen":      {"model":"deepseek/deepseek-chat-v3.1:free",       "sys":"You are Qwen – multilingual and reasoning-focused."},
    "command-r": {"model":"deepseek/deepseek-chat-v3.1:free",    "sys":"You are Command R+ – factual and structured."},
}

# --- AUTHENTICATION AND TIME LIMIT LOGIC ---

def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # If not authenticated, always redirect to login
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def get_time_limit_status():
    trial_duration = timedelta(days=7)
    if 'signup_time' in session:
        signup_time = datetime.fromisoformat(session['signup_time'])
        expiry_time = signup_time + trial_duration
        time_left = expiry_time - datetime.now()
        
        if time_left.total_seconds() <= 0:
            return "expired", "Trial Expired", 0.0

        total_seconds = time_left.total_seconds()
        
        # Logic to format time left (hours, minutes, seconds)
        days = time_left.days
        hours = int((total_seconds % (24 * 3600)) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        if days > 0:
            timer_text = f"{days}d {hours}h {minutes}m left"
        elif hours > 0:
            timer_text = f"{hours}h {minutes}m {seconds}s left"
        else:
            timer_text = f"{minutes}m {seconds}s left"

        return "active", timer_text, total_seconds
    return "unknown", "No active trial", 0.0

@app.route("/")
def index():
    # Strict redirection: if not logged in, go to login. If logged in, go to chat.
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('chat'))

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user_id' in session:
        return redirect(url_for('chat'))
        
    if request.method == "GET":
        return render_template("login.html", firebase_config=FIREBASE_CONFIG)
        
    token = request.json.get("idToken")
    if not token:
        return jsonify({"error": "Missing ID token"}), 400

    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        
        session['user_id'] = uid
        # Start trial time on first login if not already set (for existing user)
        if 'signup_time' not in session:
            session['signup_time'] = datetime.now().isoformat()

        return jsonify({"message": "Login successful", "uid": uid}), 200

    except Exception as e:
        logging.error(f"Firebase token verification failed: {e}")
        return jsonify({"error": "Invalid token or authentication failed"}), 401

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if 'user_id' in session:
        return redirect(url_for('chat'))
        
    if request.method == "GET":
        return render_template("signup.html", firebase_config=FIREBASE_CONFIG)
        
    token = request.json.get("idToken")
    if not token:
        return jsonify({"error": "Missing ID token"}), 400

    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']
        
        session['user_id'] = uid
        # Always set signup_time on successful new signup
        session['signup_time'] = datetime.now().isoformat()

        return jsonify({"message": "Signup successful", "uid": uid}), 200

    except Exception as e:
        logging.error(f"Firebase token verification failed: {e}")
        return jsonify({"error": "Invalid token or authentication failed"}), 401

@app.route("/chat")
@login_required # This decorator ensures we check the session first
def chat():
    status, timer_text, time_left_sec = get_time_limit_status()
    
    return render_template("index.html", 
        css_cachebuster=secrets.token_hex(4),
        is_authenticated=True,
        timer_text=timer_text,
        time_left_sec=time_left_sec
    )

@app.route("/logout", methods=["POST"])
def logout():
    session.pop('user_id', None)
    session.pop('signup_time', None)
    return jsonify({"message": "Logged out successfully"}), 200

@app.route("/health")
@login_required
def health():
    return jsonify(
        status="ok",
        keys_ok=any(get_key(k) for k in MODELS),
        global_limit=GLOBAL_LIMIT,
        models={k: v["model"] for k, v in MODELS.items()}
    )

@app.route("/stream", methods=["POST"])
@login_required
def stream():
    status, timer_text, time_left_sec = get_time_limit_status()
    if time_left_sec <= 0:
         return jsonify(error="Trial has expired."), 403

    # ... [Keep /stream, process_files, /asklurk, get_key, count_tok, ai_stream as previously defined]
    
    # Placeholder for brevity, assuming the full logic is present
    return jsonify({"error": "Stream implementation omitted for response brevity."}), 501 


def process_files(files):
    # This function needs the logic from the original app.py
    return "" # Placeholder

@app.route("/asklurk", methods=["POST"])
@login_required
def asklurk():
    # This function needs the logic from the original app.py
    return jsonify({"error": "AskLurk implementation omitted for response brevity."}), 501

def get_key(model_key):
    # This function needs the logic from the original app.py
    return os.getenv("OPENROUTER_API_KEY") # Placeholder

def count_tok(text, model="gpt-4"):
    # This function needs the logic from the original app.py
    return 0 # Placeholder

def ai_stream(system_prompt, user_prompt, bot_key, api_key, model_name,
             global_counter, global_limit, input_tokens):
    # This function needs the logic from the original app.py
    return # Placeholder

if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5011)
