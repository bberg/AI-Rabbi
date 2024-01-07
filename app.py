from pprint import pprint as pp
import openai
import pinecone
import pandas as pd
import tiktoken
import os
import sqlite3
import time
import random
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from flask import Flask, render_template, request, Response, stream_with_context, g, url_for, redirect, flash
from flask_httpauth import HTTPBasicAuth
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import uuid



auth = HTTPBasicAuth()

app = Flask(__name__)

db_file = 'search_logs.db'

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

def init_db():
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(db_file)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

print_all = False

openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")

df = pd.read_pickle('output-5sentence_without_embeddings.pkl')

index_name = 'midrash'
encoding = tiktoken.get_encoding("cl100k_base")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )
# connect to index
pinecone_index = pinecone.Index(index_name)
# view index stats
print(pinecone_index.describe_index_stats())
with app.app_context():
    init_db()

# to keep pinecone from dumping the database, schedule at least one request per day
def scheduled_task():
    print("making keepalive request to pinecone")
    get_relevant_sources("what is the meaning of life? keepalive test")
    pass

scheduler = BackgroundScheduler()
scheduler.add_job(func=scheduled_task, trigger="interval", minutes=1)
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

@app.teardown_appcontext
def shutdown_scheduler(exception=None):
    scheduler.shutdown()

# our user model
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@auth.verify_password
def verify_password(username, password):
    # Replace these hardcoded credentials with your own
    if username == 'admin' and password == os.getenv('LOG_PASSWORD'):
        return username
    return None

def is_request_allowed(user_id):
    # todo add exception for development using cookies
    db = get_db()

    one_day_ago = datetime.utcnow() - timedelta(days=1)
    # todo remove if cookies work and this is no longer needed

    recent_requests = db.execute(
        'SELECT COUNT(*) as cnt FROM request_logs WHERE user_id = ? AND timestamp >= ?',
        (user_id, one_day_ago)
    ).fetchone()
    if recent_requests['cnt'] >= 5:
        return False

    return True

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def search_function(query,texts,user_id):
    original_query = query
    context_plus_query = ''
    ip_address = request.remote_addr
    real_ip_address = request.environ.get('HTTP_REAL_IP', request.remote_addr)

    for t in texts:
        filename = '\nsource: '+str(t['filename'])
        segment_number = '\t segment:'+str(t['segment_number'])
        text = '\ntext:' + str(t['text']+'\n\n')
        context_plus_query += filename
        # context_plus_query += segment_number
        context_plus_query += text
        if print_all:
            yield filename
            yield segment_number
            yield text
    try:
        context_plus_query += '\n\n --- \n\n + ' + query
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a Rabbi chatbot analyzing Midrash, analyze how each of the provided source texts specifically answers the question in the style of a talmudic rabbi. Provide step-by-step reasoning to help understand how each source addresses the question - cite your sources explicitly. Identify areas within or between the texts where different or conflicting advice is provided. Using only the information from the sources provide a detailed story from a rabbinic sermon to illustrate the answer to the question for modern people. Using only the information from the sources provide a nuanced theological and philosophical explanation to the question. If your are unable to answer the question using the provided context, say 'I don't know'"}
                ,{"role": "user", "content": context_plus_query}
            ],
            temperature=0.2,
            stream=True
        )
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        total_message = ''
        # iterate through the stream of events
        for chunk in response:
            if print_all:
                pp(chunk)
            collected_chunks.append(chunk)  # save the event response
            try:
                chunk_message = chunk['choices'][0]['delta']['content']  # extract the message
            except KeyError:
                chunk_message = ''

            collected_messages.append(chunk_message)  # save the message
            query += chunk_message
            yield chunk_message


    except Exception as e:
        yield str(e)
    db = get_db()
    # print(ip_address, original_query, query)
    insert_statement = "INSERT INTO response_logs (ip_address, real_ip_address, query, collected_messages, user_id) VALUES (?, ?, ?, ?, ?)"
    data_tuple = (ip_address, real_ip_address, original_query, query, user_id,)
    print(insert_statement)
    print(data_tuple)
    db.execute(insert_statement,data_tuple)
    db.commit()
def search_function_debug(query):
    start_time = time.time()
    duration = 30  # Duration in seconds

    while time.time() - start_time < duration:
        random_number = random.randint(1, 100)  # Generate a random number between 1 and 100
        yield f"data: {random_number}\n\n"  # Yield the random number as a Server-Sent Event
        time.sleep(1)

def get_relevant_sources(topic):
    # retrieve from openai
    xq = get_embedding(text=topic)

    # get relevant contexts (including the questions)
    res = pinecone_index.query(xq, top_k=3, include_metadata=True)

    # print(res)
    # TODO then do a second pass within the top articles and identify the top 1 or two segments that pertain to the item requested

    texts = []
    for i in res['matches']:
        # print(df.loc[int(i['id']),].to_dict())
        texts.append(df.loc[int(i['id']),].to_dict())
    return texts

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['search']
    # todo remove if cookies work and this is no longer needed
    ip_address = request.remote_addr
    real_ip_address = request.environ.get('HTTP_REAL_IP', request.remote_addr)
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())

    if not is_request_allowed(user_id):
        return "You have reached the daily request limit. Per-user usage is limited for now. Please try again later."

    # insert request into the log
    db = get_db()
    insert_statement = "INSERT INTO request_logs (ip_address, real_ip_address, query, user_id) VALUES (\""+ip_address+"\",\""+real_ip_address+"\",\""+query+"\",\""+user_id+"\")"
    db.execute(insert_statement)
    db.commit()


    texts = get_relevant_sources(query)
    pp(texts)
    response = Response(stream_with_context(search_function(query,texts,user_id)), content_type='text/event-stream')
    response.set_cookie('user_id', user_id, max_age=60 * 60 * 24 * 365)
    return response

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        # here we use a static user, replace with a real database lookup
        user = User('1', 'admin', generate_password_hash(os.getenv('LOG_PASSWORD')))

        # if the user exists and the password is correct, log them in
        if user and check_password_hash(user.password, password):
            login_user(user, remember=remember)
            return redirect(request.args.get('next') or url_for('index'))

        # if the username or password is incorrect, flash an error message
        flash('Invalid username or password.')
    return redirect(url_for('view_logs'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/logs')
@auth.login_required
def view_logs():
    db = get_db()
    logs = db.execute('SELECT * FROM request_logs ORDER BY timestamp DESC').fetchall()
    return render_template('request_logs.html', logs=logs)

@app.route('/response_logs')
@auth.login_required
def view_response_logs():
    db = get_db()
    logs = db.execute('SELECT * FROM response_logs ORDER BY timestamp DESC').fetchall()
    return render_template('response_logs.html', logs=logs)



if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True, port=3000)