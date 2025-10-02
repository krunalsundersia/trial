"""
Flask + OpenRouter multi-AI chat + AskLurk best-answer picker + File Upload + Firebase Auth
AUTHENTICATED ACCESS – login / sign-up / Google / Phone-OTP
"""
import os
import json
import logging
import secrets
import re
import tempfile
import time
from datetime import datetime, timedelta
from flask import Flask, request, Response, jsonify, render_template, session, redirect, url_for
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import firebase_admin
from firebase_admin import credentials, auth
import pdfplumber

# ---------------------------------------------------------
# 1.  CONFIGURATION  –  read from env
# ---------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# Firebase client-side config (passed to HTML)
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID")
}

# Firebase Admin SDK
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH")
if FIREBASE_CRED_PATH and os.path.exists(FIREBASE_CRED_PATH):
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase Admin SDK initialized.")
    except Exception as e:
        logging.error("Firebase Admin init failed: %s", e)
else:
    logging.warning("FIREBASE_CRED_PATH missing – authentication will fail.")

# ---------------------------------------------------------
# 2.  GLOBALS
# ---------------------------------------------------------
GLOBAL_LIMIT = 300
MODELS = {
    "logic":     {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "give short answer"},
    "creative":  {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "give short answer"},
    "balanced":  {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "give short answer"},
    "gpt4o":     {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "You are GPT-4o – clear, accurate, concise."},
    "claude3":   {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "You are Claude 3 – thoughtful and precise."},
    "llama31":   {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "give short answer"},
    "mixtral":   {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "give short answer"},
    "qwen":      {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "You are Qwen – multilingual and reasoning-focused."},
    "command-r": {"model": "deepseek/deepseek-chat-v3.1:free", "sys": "You are Command R+ – factual and structured."},
}

# ---------------------------------------------------------
# 3.  AUTHENTICATION DECORATOR & HELPERS
# ---------------------------------------------------------
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def get_time_limit_status():
    trial_duration = timedelta(days=7)
    if 'signup_time' in session:
        signup_time = datetime.fromisoformat(session['signup_time'])
        expiry_time = signup_time + trial_duration
        time_left = expiry_time - datetime.now()
        if time_left.total_seconds() <= 0:
            return "expired", "Trial Expired", 0.0
        total_seconds = int(time_left.total_seconds())
        days = time_left.days
        hours = (total_seconds % (24 * 3600)) // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if days > 0:
            txt = f"{days}d {hours}h {minutes}m left"
        elif hours > 0:
            txt = f"{hours}h {minutes}m {seconds}s left"
        else:
            txt = f"{minutes}m {seconds}s left"
        return "active", txt, total_seconds
    return "unknown", "No active trial", 0.0

# ---------------------------------------------------------
# 4.  ROUTES  –  AUTH
# ---------------------------------------------------------
@app.route("/")
def index():
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
        decoded = auth.verify_id_token(token)
        uid = decoded['uid']
        session['user_id'] = uid
        if 'signup_time' not in session:
            session['signup_time'] = datetime.now().isoformat()
        return jsonify({"message": "Login successful", "uid": uid}), 200
    except Exception as e:
        logging.error("Token verification failed: %s", e)
        return jsonify({"error": "Invalid token"}), 401

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
        decoded = auth.verify_id_token(token)
        uid = decoded['uid']
        session['user_id'] = uid
        session['signup_time'] = datetime.now().isoformat()
        return jsonify({"message": "Signup successful", "uid": uid}), 200
    except Exception as e:
        logging.error("Token verification failed: %s", e)
        return jsonify({"error": "Invalid token"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    session.pop('user_id', None)
    session.pop('signup_time', None)
    return jsonify({"message": "Logged out"}), 200

# ---------------------------------------------------------
# 5.  ROUTES  –  APP
# ---------------------------------------------------------
@app.route("/chat")
@login_required
def chat():
    status, timer_text, time_left_sec = get_time_limit_status()
    return render_template("index.html",
        css_cachebuster=secrets.token_hex(4),
        is_authenticated=True,
        timer_text=timer_text,
        time_left_sec=time_left_sec
    )

@app.route("/health")
@login_required
def health():
    return jsonify(
        status="ok",
        keys_ok=any(get_key(k) for k in MODELS),
        global_limit=GLOBAL_LIMIT,
        models={k: v["model"] for k, v in MODELS.items()}
    )

# ---------------------------------------------------------
# 6.  CORE  –  STREAMING CHAT
# ---------------------------------------------------------
def get_key(model_key):
    # You can later shard keys per model; for now one global key
    return os.getenv("OPENROUTER_API_KEY")

def count_tok(text, model="gpt-4"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def ai_stream(system_prompt, user_prompt, bot_key, api_key, model_name,
              global_counter, global_limit, input_tokens):
    """
    Generator that yields SSE-style chunks:
    data: {"bot":"logic","text":"hello","tokens":12}
    ...
    data: {"bot":"logic","done":true}
    """
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        stream = client.chat.completions.create(model=model_name, messages=messages,
                                                temperature=0.7, stream=True)
        buffer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            buffer += delta
            if global_counter[0] + input_tokens > global_limit:
                yield {"bot": bot_key, "error": "Global token limit reached"}
                break
            global_counter[0] += count_tok(delta)
            yield {"bot": bot_key, "text": delta, "tokens": count_tok(delta)}
        yield {"bot": bot_key, "done": True}
    except Exception as e:
        yield {"bot": bot_key, "error": str(e)}

def process_files(files):
    """
    Accepts list of FileStorage objects (from Flask request.files).
    Returns a single concatenated string of extracted text.
    """
    text_parts = []
    for file in files:
        try:
            if file.filename.lower().endswith(".pdf"):
                with pdfplumber.open(file) as pdf:
                    text_parts.append("\n".join(page.extract_text() or "" for page in pdf.pages))
            elif file.filename.lower().endswith(".txt"):
                text_parts.append(file.read().decode("utf-8", errors="ignore"))
            else:
                # Add more parsers if needed
                text_parts.append(f"[File {file.filename} attached]")
        except Exception as e:
            text_parts.append(f"[Error reading {file.filename}: {e}]")
    return "\n\n".join(text_parts)

@app.route("/stream", methods=["POST"])
@login_required
def stream():
    status, _, time_left_sec = get_time_limit_status()
    if time_left_sec <= 0:
        return jsonify(error="Trial expired"), 403

    prompt = request.form.get("prompt", "")
    selected_models = json.loads(request.form.get("selected_models", "[]"))
    files = request.files.getlist("files")
    file_text = process_files(files)
    full_prompt = f"{prompt}\n\n{file_text}".strip()

    global_counter = [0]

    def generate():
        yield "event: start\ndata: {}\n\n"
        for model_key in selected_models:
            if model_key not in MODELS:
                continue
            cfg = MODELS[model_key]
            sys = cfg["sys"]
            model_name = cfg["model"]
            api_key = get_key(model_key)
            if not api_key:
                yield f'data: {json.dumps({"bot": model_key, "error": "No API key"})}\n\n'
                continue
            input_tokens = count_tok(sys + full_prompt)
            for payload in ai_stream(sys, full_prompt, model_key, api_key, model_name,
                                     global_counter, GLOBAL_LIMIT, input_tokens):
                yield f'data: {json.dumps(payload)}\n\n'
        yield 'data: {"overall":"done"}\n\n'

    return Response(generate(), mimetype="text/event-stream")

# ---------------------------------------------------------
# 7.  ASKLURK  –  BEST-ANSWER PICKER
# ---------------------------------------------------------
@app.route("/asklurk", methods=["POST"])
@login_required
def asklurk():
    data = request.get_json(silent=True) or {}
    answers = data.get("answers", {})
    prompt = data.get("prompt", "")
    if not answers or not prompt:
        return jsonify(error="Missing answers or prompt"), 400

    # Simple deterministic picker: longest answer wins
    best_model, best_answer = max(answers.items(), key=lambda kv: len(kv[1]))
    return jsonify(best=best_answer, best_model=best_model)

# ---------------------------------------------------------
# 8.  RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Required env-vars:
    #   FLASK_SECRET_KEY=***
    #   OPENROUTER_API_KEY=***
    #   FIREBASE_CRED_PATH=path/to/serviceAccount.json
    #   (and the six FIREBASE_* keys for client-side config)
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5011)
