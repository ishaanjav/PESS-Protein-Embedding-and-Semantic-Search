import pandas as pd
import numpy as np
import ast
import pinecone
import itertools
import random
from datetime import datetime

pinecone.init(api_key="<YOUR KEY HERE>",
              environment="us-east4-gcp")

# change the CSV name and index name as needed (run once for text and once for ESM to upsert them)
# csvName = '../text_embeddings.csv'
csvName = '../esm_embeddings.csv'
indexName = 'esm-embeddings'
columnName = 'esm_embeddings'

df = pd.read_csv(csvName)
df.dropna(subset=[columnName], inplace=True)

# Function to convert string representation of array to actual array
def string_to_array(s):
    if pd.isna(s) or s == '' or s == None or s == 'nan':  # should never be true
        return []
    return np.array(ast.literal_eval(s)).tolist()


batch_size = 100
chunks_with_index = [(df[i:i + batch_size], df.index[i:i + batch_size])
                     for i in range(0, len(df), batch_size)]


# Initialize Pinecone Index
index = pinecone.Index(indexName, pool_threads=30)
idx = 0
for chunk, index_values in chunks_with_index:
    embeddings = chunk[columnName].apply(string_to_array).tolist()
    vectors = [(str(chunk.iloc[i]['Accession']), embeddings[i],
                {"Name": str(chunk.iloc[i]['Name']),
                 "CSV Index": int(index_values[i]),
                 "Description": str(chunk.iloc[i]['Description']),
                 })
               for i in range(len(embeddings))]

    # Upsert the current chunk asynchronously and wait for completion
    index.upsert(vectors=vectors, async_req=True).get()
    if idx % 5 == 0:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        print("   =======================>>>>>>>>> ",
              time_str, idx, len(chunks_with_index))
    idx += 1

# Close the Pinecone Index
index.close()