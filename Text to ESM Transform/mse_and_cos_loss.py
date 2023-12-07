import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
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


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        cosine_loss = 1 - self.cosine_similarity(outputs, targets).mean()
        return mse_loss + cosine_loss


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# DataLoader setup
batch_size = 64  # or 32 if you prefer
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size, shuffle=False)


for learning_rate in [0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01]:
    # Create an instance of the network
    model = EmbeddingNet()

    # Loss and optimizer
    criterion = CustomLoss()
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
        model.train()
        total_train_loss = 0
        total_cosine_sim = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            cosine_sim = cosine_similarity(outputs, y_batch).mean().item()
            total_cosine_sim += cosine_sim
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        average_cosine_sim = total_cosine_sim / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_cosine_sim = 0

        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                val_cosine_sim = cosine_similarity(
                    val_outputs, y_val_batch).mean().item()
                total_val_loss += val_loss.item()
                total_val_cosine_sim += val_cosine_sim

        average_val_loss = total_val_loss / len(val_loader)
        average_val_cosine_sim = total_val_cosine_sim / len(val_loader)

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_train_loss}, Cosine Sim: {average_cosine_sim}, Val Loss: {average_val_loss}, Val Cosine Sim: {average_val_cosine_sim}')
                print("Early stopping triggered")
                break
        if epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_train_loss}, Cosine Sim: {average_cosine_sim}, Val Loss: {average_val_loss}, Val Cosine Sim: {average_val_cosine_sim}')

    torch.save(model.state_dict(), 'models/mse_and_cos_loss_' +
               str(learning_rate)+'.pth')
