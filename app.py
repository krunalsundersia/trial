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
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(32))

# Firebase configuration for frontend (using dummy placeholder values for safety)
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
            # For all routes, redirect to login if not authenticated
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
        
        if total_seconds > 2 * 24 * 3600:
            days_left = time_left.days
            return "days", f"{days_left} days left", total_seconds
        elif total_seconds > 3600:
            hours_left = int(total_seconds // 3600)
            minutes_left = int((total_seconds % 3600) // 60)
            return "hours", f"{hours_left}h {minutes_left}m left", total_seconds
        elif total_seconds > 60:
            minutes_left = int(total_seconds // 60)
            seconds_left = int(total_seconds % 60)
            return "minutes", f"{minutes_left}m {seconds_left}s left", total_seconds
        else:
            seconds_left = int(total_seconds)
            return "seconds", f"{seconds_left}s left", total_seconds
    return "unknown", "No active trial", 0.0

@app.route("/")
def index():
    # Redirect to login if not authenticated, otherwise to chat
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
@login_required
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

# --- Health Check, Stream, AskLurk, and Utility Functions (Unchanged) ---
# ... [Keeping /health, /stream, process_files, /asklurk, get_key, count_tok, ai_stream as they were in the original app.py]

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

    if request.content_type and 'multipart/form-data' in request.content_type:
        prompt = request.form.get("prompt", "").strip()
        selected_models_json = request.form.get("selected_models", "[]")
        selected = json.loads(selected_models_json)
        files = request.files.getlist('files')
    else:
        data = request.json
        prompt = data.get("prompt", "").strip()
        selected = data.get("selected_models", [])
        files = []
    
    if not prompt and not files:
        return jsonify(error="Missing prompt or files"), 400
    if not selected:
        return jsonify(error="No models selected"), 400
    
    # Only process selected models
    valid = [k for k in selected if k in MODELS]
    if not valid:
        return jsonify(error="No valid models"), 400

    def generate():
        counter = [0]
        input_tokens = count_tok(prompt)
        
        file_context = ""
        if files:
            file_context = process_files(files)
            input_tokens += count_tok(file_context)
        
        full_prompt = prompt
        if file_context:
            full_prompt = f"{prompt}\n\nAttached files context:\n{file_context}" if prompt else file_context

        # Only stream for selected models
        for key in valid:
            for chunk in ai_stream(MODELS[key]["sys"], full_prompt, key, get_key(key),
                                   MODELS[key]["model"], counter, GLOBAL_LIMIT, input_tokens):
                yield chunk
                if counter[0] >= GLOBAL_LIMIT:
                    break
        yield f"data: {json.dumps({'overall': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "Connection": "keep-alive",
                             "X-Accel-Buffering": "no"})

def process_files(files):
    file_contexts = []
    
    for file in files:
        try:
            filename = file.filename
            file_extension = os.path.splitext(filename)[1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                file.save(tmp.name)
                temp_path = tmp.name

            content = ""
            
            if file_extension == '.txt':
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            elif file_extension == '.pdf':
                try:
                    with pdfplumber.open(temp_path) as pdf:
                        content = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                except Exception as e:
                    content = f"[PDF ERROR: Failed to extract text from PDF: {e}]"
            
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                content = f"[Visual Content: This is an image file. Text extraction not implemented.]"
            
            elif file_extension in ['.doc', '.docx', '.csv', '.xlsx', '.pptx']:
                content = f"[Document Content: {file_extension.upper()} file. Full content extraction requires specialized libraries.]"
            
            else:
                content = f"[Unsupported file type: {file_extension}]"
            
            os.unlink(temp_path)

            MAX_CONTENT_LEN = 5000
            if len(content) > MAX_CONTENT_LEN:
                content = content[:MAX_CONTENT_LEN] + "\n[... Content truncated ...]"
            
            file_contexts.append(f"File: {filename} (Type: {file_extension.strip('.')})\nContent Snippet:\n{content.strip()}\n")
            
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {str(e)}")
            file_contexts.append(f"File: {filename} (Error: {str(e)})\n")
    
    return "\n".join(file_contexts)

@app.route("/asklurk", methods=["POST"])
@login_required
def asklurk():
    data = request.json
    answers = data.get("answers", {})
    prompt = data.get("prompt", "")

    if not answers:
        return jsonify(error="No answers to analyze"), 400

    try:
        valid_answers = {k: v for k, v in answers.items() if v and v.strip()}
        
        if not valid_answers:
            return jsonify(error="No valid answers to analyze"), 400

        scored_answers = []
        
        for bot_key, answer in valid_answers.items():
            score = 0
            
            word_count = len(answer.split())
            if 50 <= word_count <= 200:
                score += 3
            elif word_count > 200:
                score += 2
            else:
                score += 1
            
            if '\n\n' in answer or '- ' in answer or '* ' in answer:
                score += 2
            
            question_indicators = ['?', 'how', 'what', 'why', 'when', 'where']
            if any(indicator in prompt.lower() for indicator in question_indicators):
                if any(indicator in answer.lower() for indicator in ['because', 'therefore', 'thus', 'so']):
                    score += 2
            
            if bot_key in ['gpt4o', 'claude3']:
                score += 1
                
            scored_answers.append((answer, score, bot_key))
        
        if scored_answers:
            scored_answers.sort(key=lambda x: x[1], reverse=True)
            best_answer, best_score, best_bot = scored_answers[0]
            
            best_answer = f"🤖 {best_bot.upper()} provided the most comprehensive answer:\n\n{best_answer}"
            
            return jsonify(best=best_answer, best_model=best_bot, score=best_score)
        else:
            return jsonify(error="Could not determine best answer"), 400
            
    except Exception as e:
        logging.error(f"AskLurk analysis error: {str(e)}")
        longest_answer = max(valid_answers.values(), key=len, default="")
        return jsonify(best=f"Fallback - Longest answer:\n\n{longest_answer}")

def get_key(model_key):
    return os.getenv("OPENROUTER_API_KEY") or os.getenv(f"OPENROUTER_API_KEY_{model_key.upper()}")

def count_tok(text, model="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return int(len(text.split()) * 0.83)

def ai_stream(system_prompt, user_prompt, bot_key, api_key, model_name,
             global_counter, global_limit, input_tokens):
    if not api_key:
        err = f"API key for {bot_key} missing. Please set OPENROUTER_API_KEY environment variable."
        logging.error(err)
        yield f"data: {json.dumps({'bot': bot_key, 'error': err})}\n\n"
        return

    client = None
    stream = None
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=30.0,
            max_retries=2
        )

        if global_counter[0] >= global_limit:
            yield f"data: {json.dumps({'bot': bot_key, 'error': 'Global token limit reached'})}\n\n"
            return

        logging.info(f"Starting stream for {bot_key} with model {model_name}")
        
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            stream=True,
            extra_headers={
                "HTTP-Referer": "http://localhost:5011",
                "X-Title": "Flask Chat App"
            }
        )

        response_text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                if content:
                    response_text += content
                    output_tokens = count_tok(content)
                    global_counter[0] += output_tokens
                    yield f"data: {json.dumps({'bot': bot_key, 'text': content, 'tokens': output_tokens})}\n\n"
                    
                    if global_counter[0] >= global_limit:
                        yield f"data: {json.dumps({'bot': bot_key, 'error': 'Token limit reached'})}\n\n"
                        return

        yield f"data: {json.dumps({'bot': bot_key, 'done': True})}\n\n"
        logging.info(f"Completed stream for {bot_key}")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logging.error(f"Error in {bot_key} stream: {error_msg}")
        yield f"data: {json.dumps({'bot': bot_key, 'error': error_msg})}\n\n"
    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

if __name__ == "__main__":
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logging.warning("OPENROUTER_API_KEY environment variable not set. Some models may not work.")
    
    # You'll need to create a 'templates' folder and put the HTML files inside.
    # The default port is 5000, changed to 5011 as per original code.
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5011)
