"""
Pure Logistic Regression in PyTorch
EXACTLY like sklearn.linear_model.LogisticRegression
"""

import torch
import torch.nn as nn


class LogisticModel(nn.Module):
    """
    Pure Logistic Regression
    NO ReLU, NO Dropout - just one linear layer!
    """
    
    def __init__(self, input_features=13):
        super(LogisticModel, self).__init__()
        
        # ONLY one linear layer!
        self.fc = nn.Linear(input_features, 2)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: [batch_size, sequence_length, features]
        Returns:
            logits: [batch_size, 2]
        """
        # Take ONLY last timestep
        x = x[:, -1, :]  # [batch_size, features]
        
        # Pure linear transformation
        x = self.fc(x)
        
        return x


if __name__ == "__main__":
    # Test model
    model = LogisticModel(input_features=13)
    
    # Test input
    x = torch.randn(64, 240, 13)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")