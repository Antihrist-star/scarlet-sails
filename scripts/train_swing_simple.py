import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import os

# LOAD DATA
print("Loading data...")
X_train = torch.load('models/X_train_swing_3d.pt') if os.path.exists('models/X_train_swing_3d.pt') else None

if X_train is None:
    from prepare_swing_data import load_and_prepare_swing_data
    X_train, X_test, y_train, y_test = load_and_prepare_swing_data(3)
    
    # Save для быстрого перезапуска
    torch.save(X_train, 'models/X_train_swing_3d.pt')
    torch.save(X_test, 'models/X_test_swing_3d.pt')
    torch.save(y_train, 'models/y_train_swing_3d.pt')
    torch.save(y_test, 'models/y_test_swing_3d.pt')
else:
    X_test = torch.load('models/X_test_swing_3d.pt')
    y_train = torch.load('models/y_train_swing_3d.pt')
    y_test = torch.load('models/y_test_swing_3d.pt')

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# SIMPLE MODEL
class SimpleModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(60 * input_features, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# CPU MODE
device = torch.device("cpu")
print(f"Device: {device}")

model = SimpleModel(X_train.shape[2])
model.to(device)

# FOCAL LOSS (для imbalanced data)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# TRAINING
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

print("\nTraining...")
for epoch in range(20):
    model.train()
    epoch_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        if torch.isnan(loss):
            print(f"NaN at epoch {epoch}, skipping")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/20, Loss: {epoch_loss/len(train_loader):.4f}")

# EVALUATION
print("\nEvaluating...")
model.eval()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu())
        all_targets.append(batch_y)

predicted = torch.cat(all_preds)
y_test = torch.cat(all_targets)

accuracy = (predicted == y_test).float().mean()

up_mask = (predicted == 1)
if up_mask.sum() > 0:
    win_rate = ((predicted == 1) & (y_test == 1)).sum().float() / up_mask.sum()
else:
    win_rate = 0.0

print(f"\n{'='*60}")
print("RESULTS - 3 DAYS")
print(f"{'='*60}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"UP predictions: {up_mask.sum()}/{len(predicted)}")

if win_rate >= 0.52:
    print("✅ SUCCESS!")
elif win_rate >= 0.48:
    print("⚠️ MARGINAL")
else:
    print("❌ NOT PROFITABLE")

# SAVE
metrics = {
    'accuracy': float(accuracy),
    'win_rate': float(win_rate)
}

with open('reports/swing_3d_simple.json', 'w') as f:
    json.dump(metrics, f, indent=2)