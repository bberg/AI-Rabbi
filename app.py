from pprint import pprint as pp
import openai
import pinecone
import pandas as pd
import tiktoken
import os
import sqlite3
import time
import random
from flask import Flask, render_template, request, Response, stream_with_context, g, url_for, redirect
from flask_httpauth import HTTPBasicAuth
from datetime import datetime, timedelta
import uuid



auth = HTTPBasicAuth()

app = Flask(__name__)

db_file = 'search_logs.db'

# Update this list with the IP addresses you want to exempt
ALLOWED_IPS = ['192.0.2.1', '192.0.2.2']


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

app = Flask(__name__)
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

@auth.verify_password
def verify_password(username, password):
    # Replace these hardcoded credentials with your own
    if username == 'admin' and password == os.getenv('LOG_PASSWORD'):
        return username
    return None

def is_request_allowed(user_id):
    # todo add exception for development using cookies
    # if ip_address in ALLOWED_IPS or real_ip_address in ALLOWED_IPS:
    #     return True
    #
    db = get_db()

    one_day_ago = datetime.utcnow() - timedelta(days=1)
    # todo remove if cookies work and this is no longer needed
    # recent_requests = db.execute(
    #     'SELECT COUNT(*) as cnt FROM search_logs WHERE ip_address = ? AND timestamp >= ?',
    #     (ip_address, one_day_ago)
    # ).fetchone()
    #
    # recent_requests_real = db.execute(
    #     'SELECT COUNT(*) as cnt FROM search_logs WHERE real_ip_address = ? AND timestamp >= ?',
    #     (real_ip_address, one_day_ago)
    # ).fetchone()
    #
    # if recent_requests['cnt'] >= 5 or recent_requests_real['cnt'] >= 5:
    #     return False

    recent_requests = db.execute(
        'SELECT COUNT(*) as cnt FROM search_logs WHERE user_id = ? AND timestamp >= ?',
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
    insert_statement = "INSERT INTO search_logs (ip_address, real_ip_address, query, collected_messages, user_id) VALUES (\""+ip_address+"\",\""+real_ip_address+"\",\""+original_query+"\",\""+query+"\",\""+user_id+"\")"
    print(insert_statement)
    db.execute(insert_statement)
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
    # ip_address = request.remote_addr
    # real_ip_address = request.environ.get('HTTP_REAL_IP', request.remote_addr)
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())

    if not is_request_allowed(user_id):
        return "You have reached the daily request limit. Per-user usage is limited for now. Please try again later."

    texts = get_relevant_sources(query)
    pp(texts)
    response = Response(stream_with_context(search_function(query,texts,user_id)), content_type='text/event-stream')
    response.set_cookie('user_id', user_id, max_age=60 * 60 * 24 * 365)
    return response

@app.route('/logs')
@auth.login_required
def view_logs():
    db = get_db()
    logs = db.execute('SELECT * FROM search_logs ORDER BY timestamp DESC').fetchall()
    return render_template('logs.html', logs=logs)

if __name__ == '__main__':
    app.run(debug=True, port=3000)