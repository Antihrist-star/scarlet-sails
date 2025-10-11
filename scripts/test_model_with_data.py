import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models.hybrid_model import HybridModel

# Generate synthetic data
batch_size = 32
seq_len = 60
features = 31

X = torch.randn(batch_size, seq_len, features)
print(f"✅ Synthetic data: {X.shape}")

# Load model
model = HybridModel(input_features=features, sequence_length=seq_len)
model.eval()

# Forward pass
with torch.no_grad():
    output = model(X)

print(f"✅ Model output: {output.shape}")
print(f"✅ Output range: [{output.min():.3f}, {output.max():.3f}]")

# Check predictions
probs = torch.softmax(output, dim=1)
predictions = torch.argmax(probs, dim=1)

print(f"✅ Predictions (UP): {(predictions == 1).sum().item()}/{batch_size}")
print(f"✅ Predictions (DOWN): {(predictions == 0).sum().item()}/{batch_size}")

print("\n🎉 MODEL TEST PASSED")