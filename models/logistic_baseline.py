"""
Logistic Baseline Model with Gradient Clipping & Lower LR
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LogisticBaseline(nn.Module):
    def __init__(self, input_dim, class_weights=None):
        super(LogisticBaseline, self).__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.class_weights = class_weights
        
        # Better initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        # x: (batch, features) - already 2D!
        logits = self.linear(x)
        return logits
    
    def fit(self, X, y, epochs=100, lr=0.0001, batch_size=512):
        """
        Train model with lower LR and gradient clipping
        """
        # Convert to tensors
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()
        
        if not isinstance(y, torch.Tensor):
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = y.long()
        
        # Loss function
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer with LOWER learning rate
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        
        self.train()
        
        n_samples = len(X_tensor)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                batch_X = X_tensor[start_idx:end_idx]
                batch_y = y_tensor[start_idx:end_idx]
                
                # Forward
                logits = self.forward(batch_X)
                loss = criterion(logits, batch_y)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"⚠️  NaN detected at epoch {epoch}, batch {i}")
                    print(f"   Logits: min={logits.min():.4f}, max={logits.max():.4f}")
                    print(f"   X: min={batch_X.min():.4f}, max={batch_X.max():.4f}")
                    return  # Stop training
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # GRADIENT CLIPPING
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            
            if epoch % 20 == 0:
                print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}")
        
        print(f"Final Loss: {avg_loss:.4f}")
    
    def predict_proba(self, X, batch_size=1024):
        """Predict probabilities"""
        self.eval()
        
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = X.float()
        
        n_samples = len(X_tensor)
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_X = X_tensor[i:i+batch_size]
                logits = self.forward(batch_X)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.numpy())
        
        return np.vstack(all_probs)
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)