"""
AI-Rabbi Flask Application

A chatbot that answers life's questions using Jewish Midrash source texts.
Uses semantic search to find relevant passages and GPT-4 for synthesis.
"""

from pprint import pprint as pp
import openai
import pandas as pd
import tiktoken
import os
import sqlite3
import time
import random
from datetime import datetime, timedelta
import uuid

from flask import Flask, render_template, request, Response, stream_with_context, g, url_for, redirect, flash, jsonify
from flask_httpauth import HTTPBasicAuth
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Conditional imports based on backend
VECTOR_DB_BACKEND = os.environ.get("VECTOR_DB_BACKEND", "pinecone")

if VECTOR_DB_BACKEND == "pgvector":
    from vector_db import VectorDB
else:
    import pinecone
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit


# =============================================================================
# App Configuration
# =============================================================================

auth = HTTPBasicAuth()
app = Flask(__name__)

# Security: Use environment variable for secret key
app.secret_key = os.environ.get('SECRET_KEY', 'super secret key')
if app.secret_key == 'super secret key':
    print("WARNING: Using default secret key. Set SECRET_KEY environment variable in production!")

db_file = 'search_logs.db'
print_all = False

# API Keys
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load Midrash data
df = pd.read_pickle('output-5sentence_without_embeddings.pkl')
encoding = tiktoken.get_encoding("cl100k_base")


# =============================================================================
# Vector Database Initialization
# =============================================================================

if VECTOR_DB_BACKEND == "pgvector":
    print("Using pgvector backend")
    vector_db = VectorDB(backend='pgvector')
else:
    print("Using Pinecone backend")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = os.environ.get("PINECONE_ENV")
    index_name = 'midrash'

    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )

    pinecone_index = pinecone.Index(index_name)
    print(pinecone_index.describe_index_stats())

    # Keepalive scheduler for Pinecone (prevents database dormancy on free tier)
    def scheduled_task():
        print("Making keepalive request to Pinecone")
        get_relevant_sources("what is the meaning of life? keepalive test")

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=scheduled_task, trigger="interval", hours=12)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())


# =============================================================================
# Login Manager
# =============================================================================

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)


class User(UserMixin):
    """Simple user model for admin authentication."""

    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)


@auth.verify_password
def verify_password(username, password):
    """Verify HTTP Basic Auth credentials."""
    if username == 'admin' and password == os.getenv('LOG_PASSWORD'):
        return username
    return None


# =============================================================================
# Database Functions
# =============================================================================

def init_db():
    """Initialize SQLite database schema."""
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()


def get_db():
    """Get database connection for current request context."""
    if 'db' not in g:
        g.db = sqlite3.connect(db_file)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """Close database connection at end of request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()


# Initialize database
with app.app_context():
    init_db()


# =============================================================================
# Rate Limiting
# =============================================================================

def is_request_allowed(user_id: str) -> bool:
    """
    Check if user has remaining requests for today.

    Args:
        user_id: UUID identifying the user

    Returns:
        True if request is allowed, False if rate limited
    """
    db = get_db()
    one_day_ago = datetime.utcnow() - timedelta(days=1)

    # Use parameterized query to prevent SQL injection
    recent_requests = db.execute(
        'SELECT COUNT(*) as cnt FROM request_logs WHERE user_id = ? AND timestamp >= ?',
        (user_id, one_day_ago)
    ).fetchone()

    return recent_requests['cnt'] < 5


def get_remaining_requests(user_id: str) -> int:
    """Get the number of remaining requests for a user."""
    db = get_db()
    one_day_ago = datetime.utcnow() - timedelta(days=1)

    recent_requests = db.execute(
        'SELECT COUNT(*) as cnt FROM request_logs WHERE user_id = ? AND timestamp >= ?',
        (user_id, one_day_ago)
    ).fetchone()

    return max(0, 5 - recent_requests['cnt'])


# =============================================================================
# OpenAI Functions
# =============================================================================

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list[float]:
    """
    Get text embedding from OpenAI.

    Args:
        text: Text to embed
        model: OpenAI embedding model

    Returns:
        List of floats (1536 dimensions)
    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


# =============================================================================
# Search Functions
# =============================================================================

def get_relevant_sources(topic: str) -> list[dict]:
    """
    Find relevant Midrash sources for a given topic.

    Args:
        topic: The user's question or topic

    Returns:
        List of dicts with filename, segment_number, text
    """
    # Get embedding from OpenAI
    xq = get_embedding(text=topic)

    if VECTOR_DB_BACKEND == "pgvector":
        # Query pgvector
        results = vector_db.query(xq, top_k=3)
        texts = []
        for r in results:
            texts.append({
                'filename': r['filename'],
                'segment_number': r['segment_number'],
                'text': r['text']
            })
    else:
        # Query Pinecone
        res = pinecone_index.query(xq, top_k=3, include_metadata=True)
        texts = []
        for i in res['matches']:
            texts.append(df.loc[int(i['id']),].to_dict())

    return texts


def search_function(query: str, texts: list[dict], user_id: str):
    """
    Generate streaming response from GPT-4 based on Midrash sources.

    Args:
        query: User's question
        texts: Relevant source texts
        user_id: User identifier for logging

    Yields:
        Chunks of the GPT-4 response
    """
    original_query = query
    context_plus_query = ''
    ip_address = request.remote_addr
    real_ip_address = request.environ.get('HTTP_REAL_IP', request.remote_addr)

    # Build context from sources
    for t in texts:
        filename = '\nsource: ' + str(t['filename'])
        segment_number = '\t segment:' + str(t['segment_number'])
        text = '\ntext:' + str(t['text'] + '\n\n')
        context_plus_query += filename
        context_plus_query += text

        if print_all:
            yield filename
            yield segment_number
            yield text

    try:
        context_plus_query += '\n\n --- \n\n + ' + query

        # System prompt for the AI Rabbi
        system_prompt = """You are a Rabbi chatbot analyzing Midrash. Your role is to:

1. **Analyze Sources**: Examine how each of the provided source texts specifically answers the question in the style of a Talmudic rabbi.

2. **Step-by-Step Reasoning**: Provide clear reasoning to help understand how each source addresses the question. Cite your sources explicitly.

3. **Identify Tensions**: Point out areas within or between the texts where different or conflicting advice is provided.

4. **Modern Story**: Using only the information from the sources, provide a detailed story from a rabbinic sermon to illustrate the answer for modern people.

5. **Theological Explanation**: Using only the information from the sources, provide a nuanced theological and philosophical explanation.

If you are unable to answer the question using the provided context, say "I don't know" - do not make up information not contained in the sources."""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context_plus_query}
            ],
            temperature=0.2,
            stream=True,
            request_timeout=120  # 2 minute timeout
        )

        collected_messages = []

        for chunk in response:
            if print_all:
                pp(chunk)

            try:
                chunk_message = chunk['choices'][0]['delta']['content']
            except KeyError:
                chunk_message = ''

            collected_messages.append(chunk_message)
            query += chunk_message
            yield chunk_message

    except openai.error.Timeout:
        yield "\n\n*The request timed out. Please try again.*"
    except openai.error.APIError as e:
        yield f"\n\n*An error occurred: {str(e)}*"
    except Exception as e:
        yield f"\n\n*An unexpected error occurred: {str(e)}*"

    # Log the response (using parameterized query)
    db = get_db()
    insert_statement = """
        INSERT INTO response_logs (ip_address, real_ip_address, query, collected_messages, user_id)
        VALUES (?, ?, ?, ?, ?)
    """
    db.execute(insert_statement, (ip_address, real_ip_address, original_query, query, user_id))
    db.commit()


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """Render the main search page."""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """
    Handle search requests.

    Returns streaming response with GPT-4 generated content.
    Returns 429 status if rate limited.
    """
    query = request.form.get('search', '').strip()

    if not query:
        return "Please enter a question", 400

    ip_address = request.remote_addr
    real_ip_address = request.environ.get('HTTP_REAL_IP', request.remote_addr)
    user_id = request.cookies.get('user_id')

    if not user_id:
        user_id = str(uuid.uuid4())

    # Check rate limit
    if not is_request_allowed(user_id):
        response = Response(
            "You have reached the daily request limit. Per-user usage is limited for now. Please try again later.",
            status=429,
            mimetype='text/plain'
        )
        response.set_cookie('user_id', user_id, max_age=60 * 60 * 24 * 365)
        return response

    # Log the request (using parameterized query - SQL injection fix)
    db = get_db()
    insert_statement = """
        INSERT INTO request_logs (ip_address, real_ip_address, query, user_id)
        VALUES (?, ?, ?, ?)
    """
    db.execute(insert_statement, (ip_address, real_ip_address, query, user_id))
    db.commit()

    # Get relevant sources and stream response
    texts = get_relevant_sources(query)
    if print_all:
        pp(texts)

    response = Response(
        stream_with_context(search_function(query, texts, user_id)),
        content_type='text/event-stream'
    )
    response.set_cookie('user_id', user_id, max_age=60 * 60 * 24 * 365)
    return response


@app.route('/api/remaining-requests')
def remaining_requests():
    """API endpoint to check remaining requests for the day."""
    user_id = request.cookies.get('user_id')
    if not user_id:
        return jsonify({'remaining': 5, 'limit': 5})

    remaining = get_remaining_requests(user_id)
    return jsonify({'remaining': remaining, 'limit': 5})


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle admin login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User('1', 'admin', generate_password_hash(os.getenv('LOG_PASSWORD', '')))

        if user and check_password_hash(user.password, password):
            login_user(user, remember=remember)
            return redirect(request.args.get('next') or url_for('index'))

        flash('Invalid username or password.')

    return redirect(url_for('view_logs'))


@app.route('/logout')
def logout():
    """Handle admin logout."""
    logout_user()
    return redirect(url_for('index'))


@app.route('/logs')
@auth.login_required
def view_logs():
    """View request logs (admin only)."""
    db = get_db()
    logs = db.execute('SELECT * FROM request_logs ORDER BY timestamp DESC').fetchall()
    return render_template('request_logs.html', logs=logs)


@app.route('/response_logs')
@auth.login_required
def view_response_logs():
    """View response logs (admin only)."""
    db = get_db()
    logs = db.execute('SELECT * FROM response_logs ORDER BY timestamp DESC').fetchall()
    return render_template('response_logs.html', logs=logs)


@app.route('/health')
def health():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'backend': VECTOR_DB_BACKEND,
        'timestamp': datetime.utcnow().isoformat()
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=3000)
