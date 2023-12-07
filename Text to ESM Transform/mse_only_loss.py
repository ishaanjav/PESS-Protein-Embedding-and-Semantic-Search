import pandas as pd
from sklearn.linear_model import LinearRegression
import ast
from sklearn.model_selection import train_test_split
from torch.nn.functional import cosine_similarity
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import euclidean
import torch.optim as optim
import sys

learning_rate = 0.001
if len(sys.argv) > 1:
    option = float(sys.argv[1])

# Load the DataFrames
df_text = pd.read_csv('../text_embeddings.csv')
df_esm = pd.read_csv('../esm_embeddings.csv')

# Preprocess: Drop rows with NaN in 'Description'
df_text = df_text.dropna(subset=['Description'])

# Ensure alignment
df_esm = df_esm.iloc[df_text.index]

# Convert embeddings from string to lists
print(len(df_esm))
df_text['ada_embedding'] = df_text['ada_embedding'].apply(ast.literal_eval)
df_esm['esm_embeddings'] = df_esm['esm_embeddings'].apply(ast.literal_eval)

X = torch.tensor(df_text['ada_embedding'].tolist()).float()
y = torch.tensor(df_esm['esm_embeddings'].tolist()).float()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.06724, random_state=42)

# Define the neural network


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


for learning_rate in [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01]:
    # Create an instance of the network
    model = EmbeddingNet()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    early_stopping_patience = 3
    if learning_rate > 0.005:
        early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_loss = float('inf')

    # Training loop
    num_epochs = 200  # Adjust as needed
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        cosine_sim = cosine_similarity(outputs, y_train).mean().item()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_cosine_sim = cosine_similarity(
                val_outputs, y_val).mean().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Cosine Sim: {cosine_sim}, Val Loss: {val_loss}, Val Cosine Sim: {val_cosine_sim}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    torch.save(model.state_dict(), 'models/mse_only_loss_' +
               str(learning_rate)+'.pth')
