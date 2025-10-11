import torch
import torch.nn as nn


class HybridModel(nn.Module):
    """CNN-LSTM Hybrid for swing trading prediction"""
    
    def __init__(self, input_features=31, sequence_length=60):
        super().__init__()
        
        # CNN Block - local pattern extraction
        self.cnn_block = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(sequence_length // 2)
        )
        
        # LSTM Block - temporal memory
        self.lstm = nn.LSTM(
            input_size=32,           # CNN output channels
            hidden_size=64,          # LSTM hidden size
            num_layers=2,            # 2 layers for depth
            batch_first=True,        # (batch, seq, features)
            dropout=0.3,             # Between layers
            bidirectional=False      # CRITICAL: no future info
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(64, 32),       # LSTM hidden → FC
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)         # Binary: [DOWN, UP]
        )
    
    def forward(self, x):
        """
        Forward pass through CNN → LSTM → Output
        
        Args:
            x: (batch, seq_len, features) - input sequences
            
        Returns:
            output: (batch, 2) - logits for [DOWN, UP]
        """
        batch_size = x.size(0)
        
        # CNN expects: (batch, channels, seq_len)
        x = x.transpose(1, 2)  # → (batch, features, seq_len)
        
        # CNN feature extraction
        x = self.cnn_block(x)  # → (batch, 32, seq_len//2)
        
        # LSTM expects: (batch, seq_len, features)
        x = x.transpose(1, 2)  # → (batch, seq_len//2, 32)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        # hidden: (num_layers, batch, hidden_size)
        
        # Use last layer's hidden state
        final_hidden = hidden[-1]  # → (batch, 64)
        
        # Output classification
        output = self.fc(final_hidden)  # → (batch, 2)
        
        return output


if __name__ == "__main__":
    # Test model
    model = HybridModel(input_features=31, sequence_length=60)
    x = torch.randn(4, 60, 31)
    output = model(x)
    print(f"✅ Model test: {x.shape} → {output.shape}")
    print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")