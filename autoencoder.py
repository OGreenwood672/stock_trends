import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GLOBALS import NORMALISED_STOCKS_PARQUET, AUTOENCODER_MODEL

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def prepare_data(file_path, sequence_length=10):
    # Read parquet file
    df = pd.read_parquet(file_path)
        
    # Create sequences
    sequences = []
    values = df.values
    for i in range(len(df) - sequence_length + 1):
        sequences.append(values[i:i+sequence_length])
    
    return torch.FloatTensor(sequences)

def train_autoencoder(X, model, num_epochs=100, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")
    X = X.to(device)
    model.to(device)
    
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            data = batch[0]
            
            # Flatten the input
            batch_size, seq_len, features = data.shape
            data_flat = data.reshape(batch_size, seq_len * features)
            
            optimizer.zero_grad()
            output = model(data_flat)
            loss = criterion(output, data_flat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def create_autoencoder(file_path, sequence_length=10):
    # Prepare data
    X = prepare_data(file_path, sequence_length)
    
    # Create model
    input_dim = X.shape[1] * X.shape[2]  # sequence_length * num_features
    model = Autoencoder(input_dim)
    
    # Train model
    losses = train_autoencoder(X, model, num_epochs=100)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    torch.save(model.state_dict(), AUTOENCODER_MODEL)

if __name__ == "__main__":
    model, scaler = create_autoencoder(NORMALISED_STOCKS_PARQUET, sequence_length=10)
    torch.save(model.state_dict(), AUTOENCODER_MODEL)