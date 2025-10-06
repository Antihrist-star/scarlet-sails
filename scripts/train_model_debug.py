import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from prepare_data_v3 import load_and_preprocess_data_v3

class ImprovedCNN1D(nn.Module):
    def __init__(self, input_channels):
        super(ImprovedCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        # После 3 pooling: 100 -> 50 -> 25 -> 12
        self.fc1 = nn.Linear(32 * 12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def train_improved_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Увеличенный batch size
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = ImprovedCNN1D(X.shape[2])
    criterion = nn.BCEWithLogitsLoss()
    
    # Увеличенный learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    print(f"Training on device: {device}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    best_accuracy = 0
    for epoch in range(50):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_X = batch_X.permute(0, 2, 1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        # Validation каждые 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                X_test_permuted = X_test.permute(0, 2, 1)
                outputs = model(X_test_permuted)
                predictions = torch.sigmoid(outputs) > 0.5
                accuracy = (predictions.squeeze() == y_test).float().mean()
                
                print(f"Epoch [{epoch+1}/50], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), "models/best_cnn_model.pth")
    
    print(f"Best Accuracy: {best_accuracy:.4f}")
    return model

if __name__ == "__main__":
    X, y, scaler_X, features = load_and_preprocess_data_v3(symbol="BTC/USDT")
    model = train_improved_model(X, y)