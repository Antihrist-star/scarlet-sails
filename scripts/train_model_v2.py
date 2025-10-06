import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
from prepare_data_v3 import load_and_preprocess_data_v3

class CNN1D(nn.Module):
    def __init__(self, input_channels, output_size):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc_input_size = self._get_conv_output_size(input_channels)
        self.fc1 = nn.Linear(self.fc_input_size, 50)
        self.fc2 = nn.Linear(50, output_size)
    
    def _get_conv_output_size(self, input_channels):
        dummy_input = torch.randn(1, input_channels, 100)  # Исправлено
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
        return x  # Убран sigmoid

def train_model(X, y, model_path="models/1d_cnn_model.pth", epochs=20, batch_size=32, learning_rate=0.001):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    input_channels = X.shape[2]
    output_size = 1
    model = CNN1D(input_channels, output_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    print(f"Using device: {device}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = batch_X.permute(0, 2, 1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Test accuracy
    model.eval()
    with torch.no_grad():
        X_test_permuted = X_test.permute(0, 2, 1)
        outputs = model(X_test_permuted)
        predictions = torch.sigmoid(outputs) > 0.5
        accuracy = (predictions.squeeze() == y_test).float().mean()
        print(f"Test Accuracy: {accuracy:.4f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    X, y, scaler_X, features = load_and_preprocess_data_v3(symbol="BTC/USDT")  # Исправлено
    print(f"Training on {len(features)} features: {features}")
    trained_model = train_model(X, y)
    print("Model training complete.")