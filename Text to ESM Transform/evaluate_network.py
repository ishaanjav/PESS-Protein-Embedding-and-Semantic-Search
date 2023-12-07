import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast
import sys
from scipy.spatial.distance import euclidean


# You can provide a list of model files to evaluate
model_files = ['mse_only_loss_0.003.pth',
               'cos_only_loss_0.005.pth']
option = 0
if len(sys.argv) > 1:
    option = int(sys.argv[1])


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(1536, 768)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(768, 1280)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load the PyTorch model
model = EmbeddingNet()
model.load_state_dict(torch.load('models/' + model_files[option]))
model.eval()

# Load and preprocess DataFrames
df_text = pd.read_csv('../text_embeddings.csv').head(100)
df_esm = pd.read_csv('../esm_embeddings.csv').head(100)
df_text = df_text.dropna(subset=['Description'])
df_esm = df_esm.iloc[df_text.index]
df_text['ada_embedding'] = df_text['ada_embedding'].apply(ast.literal_eval)
df_esm['esm_embeddings'] = df_esm['esm_embeddings'].apply(ast.literal_eval)

# Select examples and perform inference
samples = 25
examples = df_text['ada_embedding'].iloc[:samples].tolist()
actual_esm_embeddings = df_esm['esm_embeddings'].iloc[:samples].tolist()

for i, example in enumerate(examples):
    example_tensor = torch.tensor(example).float().unsqueeze(0)
    with torch.no_grad():
        transformed_embedding = model(example_tensor).numpy().flatten()
    actual_embedding = actual_esm_embeddings[i]

    # Compute metrics
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
