import pandas as pd
from sklearn.linear_model import LinearRegression
import ast
import numpy as np

# Load the DataFrames
df_text = pd.read_csv('text_embeddings.csv')
df_esm = pd.read_csv('esm_embeddings.csv')

# Preprocess: Drop rows with NaN in 'Description'
df_text = df_text.dropna(subset=['Description'])

# Ensure alignment
df_esm = df_esm.iloc[df_text.index]

# Convert embeddings from string to lists
print(len(df_esm))
print(df_text.head())
print(df_esm.head())
df_text['ada_embedding'] = df_text['ada_embedding'].apply(ast.literal_eval)
df_esm['esm_embeddings'] = df_esm['esm_embeddings'].apply(ast.literal_eval)

# Prepare the data for regression
X = np.array(df_text['ada_embedding'].tolist())  # Text embeddings
y = np.array(df_esm['esm_embeddings'].tolist())  # ESM embeddings

print(X.shape)
print(len(df_esm))

reg = LinearRegression(fit_intercept=False)
reg.fit(X, y)

# Get W from the coefficients
W = reg.coef_.T

print(W.shape)
# Save W
np.save('V.npy', W)

# Inverting W (note: this might not always be possible if W is not square or singular)
W_pseudoinv = np.linalg.pinv(W)

# Save the pseudoinverse
np.save('V_pseudoinv.npy', W_pseudoinv)
