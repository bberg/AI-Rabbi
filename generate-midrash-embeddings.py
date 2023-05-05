import openai
import os
import fnmatch
import tiktoken
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import time
from pprint import pprint as pp
import pinecone
import pandas as pd


nltk.download('punkt')

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def find_text_files(directory):
    text_files = []
    for root, dirs, files in os.walk(directory):
        if root.split('/')[-1] == 'English':
            for file in files:
                if fnmatch.fnmatch(file, '*.txt'):
                    text_files.append(os.path.join(root, file))
    return text_files

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def create_dataframe(text_files,sentence_block_length):
    data = []
    for file in text_files:
        content = read_text_file(file)
        sentences = sent_tokenize(content)

        # Split the text into segments of sentences
        segments = [sentences[i:i + sentence_block_length] for i in range(0, len(sentences), sentence_block_length)]

        for idx, segment in enumerate(segments):
            segment_text = " ".join(segment)
            segment_text = segment_text.replace('\n', ' ').replace('\r', ' ')
            data.append([file, idx + 1, segment_text])

    df = pd.DataFrame(data, columns=['filename', 'segment_number', 'text'])
    return df

# gets emeddings for the column 'text'
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    result = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    return result

def text_stats(text_file_list):
    total = ''
    for t in text_files:
        print(t)
        total += read_text_file(t)

    total_chars = len(total)
    estimated_tokens = total_chars / 4.05
    actual_tokens = num_tokens_from_string(total, "cl100k_base")
    thousand_tokens = actual_tokens / 1000
    estimated_cost_to_embed = 0.0004 * thousand_tokens

    print('total files: ', len(text_files))
    print("estimated_tokens", estimated_tokens)
    print("actual_tokens", actual_tokens)
    # print('total segments: ',thousand_tokens)
    print("estimated_cost_to_embed: ", estimated_cost_to_embed)

def initialize_pinecone(pinecone_api_key,pinecone_env,pinecone_index_name):
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    # check if index already exists (it shouldn't if this is first time)
    if pinecone_index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            pinecone_index_name,
            dimension=1536,
            metric='cosine',
            metadata_config={'indexed': ['channel_id', 'published']}
        )
    # connect to index
    index = pinecone.Index(pinecone_index_name)
    # view index stats
    print(index.describe_index_stats())
    return index



# CONFIG
# todo move keys to a safe location

openai.api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")

pinecone_index_name = 'midrash'

# directory containing text files
directory = 'Sefaria-Export/txt/Midrash'

output_csv = 'output.csv'
output_pkl = 'output.pkl'

# how many sentences should each segment be? Total needs to be less than the token limit for ada
sentence_block_length = 25
# todo remove
# 25 -> 51828
# 50 -> 26070
# 100 -> 13182
# 1000 -> 1662


if not os.path.exists(output_pkl):

    # find the text files
    text_files = find_text_files(directory)

    # print some stats about the text files
    text_stats(text_files)

    # create a dataframe from the text files
    df = create_dataframe(text_files,sentence_block_length)

    print("numer of segments: ", df.shape[0])

    # allows us to simply shorten the length of the dataframe for testing purposes
    # df_test = df.head(10)
    df_run = df
    df_run['ada_embedding'] = np.empty((len(df_run), 0)).tolist()
    df_run['uploaded_to_pinecone'] = False
    df_run.to_csv(output_csv, index=False)
    df_run.to_pickle(output_pkl)

else:
    print("LOADING FROM FILE: ",output_pkl)
    df_run = pd.read_pickle(output_pkl)

pinecone_index = initialize_pinecone(pinecone_api_key,pinecone_env,pinecone_index_name)


batch_size = 100
batch_count = 0
for index, row in df_run.iterrows():
    try:
        if row['ada_embedding'] == []:
            batch_count+=1
            print(index)
            try:
                df_run.at[index,'ada_embedding'] = get_embedding(row['text'])
            except Exception as e:
                print(e)
                print("retrying with first 6000 characters of text only")
                df_run.at[index, 'ada_embedding'] = get_embedding(row['text'][:6000])
        if batch_count > batch_size:
            print('batch upload')
            batch = df_run[index-batch_size:index]
            # get tuples formatted for pinecone
            batch_tuples = [(str(batch_index), batch_row['ada_embedding']) for batch_index, batch_row in batch.iterrows()]
            # pp(batch_tuples)
            # upload to pinecone
            res = pinecone_index.upsert(vectors=batch_tuples)
            # update the uploaded to pinecone column
            df_run.loc[index - batch_size:index, 'uploaded_to_pinecone'] = True
            # save the csv off so we don't lose progress
            df_run.to_csv(output_csv, index=False)
            df_run.to_pickle(output_pkl)
            df_smaller = df_run.drop(columns=['ada_embedding'])
            df_smaller.to_pickle(output_pkl.split('.')[0]+'_without_embeddings.pkl')
            print('upload success')
            batch_count = 0
    except Exception as e:
        print("ERROR")
        print(e)