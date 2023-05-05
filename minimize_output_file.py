# todo [deprecated] remove
import pandas as pd

original = pd.read_pickle('output.pkl')
original_smaller = original.drop(columns=['ada_embedding'])
original_smaller.to_pickle('output_without_embeddings.pkl')