import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import ast

W = np.load('V.npy')

# Load DataFrames
df_text = pd.read_csv('text_embeddings.csv').head(100)
df_esm = pd.read_csv('esm_embeddings.csv').head(100)

# Preprocess DataFrames
df_text = df_text.dropna(subset=['Description'])
df_esm = df_esm.iloc[df_text.index]
df_text['ada_embedding'] = df_text['ada_embedding'].apply(ast.literal_eval)
df_esm['esm_embeddings'] = df_esm['esm_embeddings'].apply(ast.literal_eval)

# Select 25 examples or all
samples = 25
examples = df_text['ada_embedding'].iloc[:samples].tolist()
actual_esm_embeddings = df_esm['esm_embeddings'].iloc[:samples].tolist()

# Apply transformation and compute distances
for i, example in enumerate(examples):
    transformed_embedding = np.dot(example, W)
    actual_embedding = actual_esm_embeddings[i]

    # Compute distance
    distance = euclidean(transformed_embedding, actual_embedding)
    magnitude_squared = np.dot(actual_embedding, actual_embedding)
    dot_product = np.dot(transformed_embedding, actual_embedding)
    mae_per_entry = np.mean(np.abs(transformed_embedding - actual_embedding))

    transformed_magnitude = np.sqrt(
        np.dot(transformed_embedding, transformed_embedding))
    actual_magnitude = np.sqrt(magnitude_squared)
    cosine_similarity = dot_product / \
        (transformed_magnitude * actual_magnitude)

    print(f'Ex {i+1}:  Dist: {distance}  MAE: {mae_per_entry}')
    print(f'\tDots: {dot_product}  vs  {magnitude_squared}')
    print(f'\tCosine Similarity: {cosine_similarity}')
