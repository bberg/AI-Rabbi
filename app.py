from flask import Flask, render_template, request, Response, stream_with_context
from pprint import pprint as pp
import openai
import pinecone
import time
import random
import pandas as pd
import os

print_all = True

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")

df = pd.read_pickle('output-5sentence_without_embeddings.pkl')

index_name = 'midrash'

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


def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def search_function(query,texts):
    context_plus_query = ''

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
                ,{"role": "user", "content": "continue"}
            ],
            temperature=0.2,
            stream=True
        )
        # create variables to collect the stream of chunks
        collected_chunks = []
        collected_messages = []
        # iterate through the stream of events
        for chunk in response:
            pp(chunk)
            collected_chunks.append(chunk)  # save the event response
            try:
                chunk_message = chunk['choices'][0]['delta']['content']  # extract the message
            except KeyError:
                chunk_message = ''

            collected_messages.append(chunk_message)  # save the message
            # print(chunk_message)
            yield chunk_message
    except Exception as e:
        yield str(e)

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

    print(res)
    # TODO then do a second pass within the top articles and identify the top 1 or two segments that pertain to the item requested

    texts = []
    for i in res['matches']:
        # print(df.loc[int(i['id']),].to_dict())
        texts.append(df.loc[int(i['id']),].to_dict())
    return texts


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['search']
    texts = get_relevant_sources(query)
    pp(texts)
    return Response(stream_with_context(search_function(query,texts)), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)




