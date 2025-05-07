import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GLOBALS import NORMALISED_STOCKS_PARQUET, AUTOENCODER_MODEL, SHAPE_FEATURES
from get_data import get_loaded_ticketers

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


def prepare_data(file_path):
    # Read parquet file
    df = pd.read_parquet(file_path)

    # Replace NaN values with 0
    df = df.fillna(0)

    return torch.FloatTensor(df.values)

def train_autoencoder(X, model, num_epochs=100, batch_size=32):

    # CUDA ISSUES:
    # Not too sure what the issue is, code runs on CPU not GPU
    # Potentially could be GPU memory issues
    # Make batch_size=1, that fixes a issue

    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))

    X = X.to(device)
    model.to(device)
    X = X.T

    dataset = TensorDataset(X)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            data = batch[0]
                        
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def create_autoencoder(file_path):
    # Prepare data
    X = prepare_data(file_path)
    
    # Create model
    input_dim =  X.shape[0]  # sequence_length * num_features
    model = Autoencoder(input_dim)
    
    # Train model
    losses = train_autoencoder(X, model, num_epochs=300)
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    torch.save(model.state_dict(), AUTOENCODER_MODEL)

def load_autoencoder(model_path, input_dim):
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path))

    return model

def get_encoded(model, data):

    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(data)
    return encoded_data

def save_features(file_path, save_file):

    X = prepare_data(file_path)

    # Load the model and get encoded data
    model = load_autoencoder(AUTOENCODER_MODEL, input_dim=X.shape[0])
    encoded_data = get_encoded(model, X.T)

    ticketers = get_loaded_ticketers()

    # Create column names for the encoded features
    columns = ['ticketer'] + [f'encoded_{i}' for i in range(8)]

    # Create a list of rows where each row contains the ticketer and its encoded features
    save_data = [[ticker] + encoded.tolist() for ticker, encoded in zip(ticketers, encoded_data)]

    # Convert to DataFrame
    df = pd.DataFrame(save_data, columns=columns)
    df.to_csv(save_file, index=False)
    


if __name__ == "__main__":
    
    # # Example usage
    # file_path = NORMALISED_STOCKS_PARQUET

    # X = prepare_data(file_path)

    # # Load the model and get encoded data
    # model = load_autoencoder(AUTOENCODER_MODEL, input_dim=X.shape[0])
    # encoded_data = get_encoded(model, X.T)

    # print(encoded_data.shape)
    # print(X.shape)

    save_features(NORMALISED_STOCKS_PARQUET, SHAPE_FEATURES)
