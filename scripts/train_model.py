
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np

from prepare_data import load_and_preprocess_data

# Define the 1D-CNN model
class CNN1D(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        # Calculate the size of the flattened layer dynamically
        # This requires a forward pass with a dummy input
        self.fc_input_size = self._get_conv_output_size(input_channels)
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def _get_conv_output_size(self, input_channels):
        # Create a dummy input to calculate the output size of conv layers
        dummy_input = torch.randn(1, input_channels, 60) # batch_size, channels, sequence_length
        x = self.conv1(dummy_input)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.flatten().size(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def train_model(X, y, model_path="models/1d_cnn_model.pth", epochs=50, batch_size=64, learning_rate=0.001):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    input_channels = X.shape[2] # Number of features
    output_size = 1 # Predicting a single value (next close price)
    model = CNN1D(input_channels, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = batch_X.permute(0, 2, 1) # Reshape for Conv1d: (batch_size, channels, sequence_length)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model

if __name__ == "__main__":
    X, y, _ = load_and_preprocess_data()
    trained_model = train_model(X, y)
    print("Model training complete.")

