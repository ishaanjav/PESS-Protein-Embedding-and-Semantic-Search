# Messing around with esm model and learning how to use it
import matplotlib.pyplot as plt
import torch
import esm
from datetime import datetime
import pandas as pd

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


def embed(sequence):
    data = [("protein1", sequence),]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate sequence representation via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    return token_representations[0, 1: len(sequence) + 1].mean(0).detach().numpy().tolist()


cnt = 0


def get_embedding(sequence):
    global cnt
    # check if protein is missing description
    if pd.isna(sequence) or sequence == '' or sequence == None or sequence == 'nan':
        return "nan"

    # clean text
    sequence = sequence.strip('\"')
    sequence = sequence.strip('\'')
    sequence = sequence.strip('\n')
    sequence = sequence.strip('\t')
    sequence = sequence.strip(' ')

    sequence = sequence[:2000]
    embedding = embed(sequence)

    cnt += 1
    if cnt % 100 == 1:
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        print(time_str, cnt)
    return str(embedding)


df = pd.read_csv('data.csv')

df['esm_650M_embedding'] = df['Sequence'].apply(
    lambda x: get_embedding(x))
df.to_csv('../esm_embeddings.csv', index=False)
