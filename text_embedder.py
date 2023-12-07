from openai import OpenAI
from datetime import datetime
import pandas as pd
client = OpenAI(api_key="<YOUR KEY HERE>")

cnt = 0

# return embedding vector given text description.
def get_embedding(text, model="text-embedding-ada-002"):
    global cnt
    # check if protein is missing description
    if pd.isna(text) or text == '' or text == None or text == 'nan':
        return "nan"

    # clean text
    text = text.strip('\"')
    text = text.strip('\'')
    text = text.replace("\n", " ")

    embedding = client.embeddings.create(
        input=text, model=model).data[0].embedding

    cnt += 1
    if cnt % 1000 == 0:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        print(time_str, cnt)
    return str(embedding)


df = pd.read_csv('data.csv')

df['ada_embedding'] = df['Description'].apply(
    lambda x: get_embedding(x, model='text-embedding-ada-002'))
df.to_csv('text_embeddings.csv', index=False)
