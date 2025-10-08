import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import json

class CNN1D(nn.Module):
    def __init__(self, input_features, sequence_length=60):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 15, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    import sys
    sys.path.append('scripts')
    from prepare_data_v4 import load_and_preprocess_data

    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(symbol="BTC_USDT")
    print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

    input_features = X_train.shape[2]
    model = CNN1D(input_features, sequence_length=60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Balanced pos_weight для daily direction
    pos_weight = torch.tensor([1.4]).to(device)
    print(f"Using pos_weight: {pos_weight.item():.1f}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.to(device)
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    print(f"Using device: {device}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {input_features}")

    # Training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print("\nTraining with DAILY DIRECTION TARGET (v4 multi-timeframe)...")
    
    for epoch in range(30):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/30], Loss: {avg_loss:.4f}")

    # ИСПРАВЛЕНО: Батчированная evaluation для избежания OOM
    print("\n=== BATCHED EVALUATION (to avoid OOM) ===")
    model.eval()
    
    with torch.no_grad():
        # Создаем DataLoader для test set
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        
        print(f"Evaluating {len(X_test):,} samples in {len(test_loader)} batches...")
        
        # Собираем все outputs по батчам
        all_outputs = []
        for i, (batch_X, _) in enumerate(test_loader):
            batch_outputs = model(batch_X)
            all_outputs.append(batch_outputs)
            if (i + 1) % 10 == 0:
                print(f"  Processed batch {i+1}/{len(test_loader)}...")
        
        # Объединяем все outputs
        outputs = torch.cat(all_outputs, dim=0)
        probabilities = torch.sigmoid(outputs)
        
        print("Evaluation complete!\n")
        
        # Test different thresholds
        thresholds = [0.5, 0.55, 0.6]
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).float()
            
            tp = ((predictions.squeeze() == 1) & (y_test == 1)).sum().item()
            fp = ((predictions.squeeze() == 1) & (y_test == 0)).sum().item()
            tn = ((predictions.squeeze() == 0) & (y_test == 0)).sum().item()
            fn = ((predictions.squeeze() == 0) & (y_test == 1)).sum().item()
            
            accuracy = (predictions.squeeze() == y_test).float().mean().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"--- Threshold: {threshold} ---")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
            print(f"Recall: {recall:.4f} ({recall*100:.1f}%)")
            print(f"F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                    'accuracy': accuracy, 'precision': precision,
                    'recall': recall, 'f1': f1
                }

    # Final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Best Threshold: {best_threshold})")
    print(f"{'='*60}")
    
    m = best_metrics
    naive_acc = (y_test == 0).float().mean().item()
    
    print(f"\n📊 PERFORMANCE:")
    print(f"Accuracy: {m['accuracy']*100:.2f}%")
    print(f"Naive Baseline: {naive_acc*100:.2f}%")
    improvement = (m['accuracy'] - naive_acc)*100
    print(f"Improvement: {improvement:+.2f}%")
    
    print(f"\n📈 CLASSIFICATION METRICS:")
    print(f"Precision: {m['precision']*100:.1f}% (of predicted UP, how many correct)")
    print(f"Recall: {m['recall']*100:.1f}% (of actual UP, how many found)")
    print(f"F1 Score: {m['f1']:.4f}")
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print(f"               Predicted")
    print(f"            UP      DOWN")
    print(f"Actual UP   {m['tp']:5d}   {m['fn']:5d}")
    print(f"      DOWN  {m['fp']:5d}   {m['tn']:5d}")
    
    # Trading viability
    print(f"\n💰 TRADING VIABILITY:")
    if m['tp'] + m['fp'] > 0:
        win_rate = m['tp'] / (m['tp'] + m['fp'])
        print(f"Win Rate: {win_rate*100:.1f}%")
        
        # Commission impact
        commission = 0.002  # 0.2% round trip
        breakeven_wr = 0.5 + (commission/2)
        print(f"Breakeven Win Rate (with commission): {breakeven_wr*100:.1f}%")
        
        if win_rate > 0.55:
            print("✅ POTENTIALLY PROFITABLE (win rate >55%)")
        elif win_rate > breakeven_wr:
            print("⚠️ MARGINAL (barely above breakeven)")
        else:
            print("❌ NOT PROFITABLE (below breakeven)")
        
        # Expected value
        avg_win = 0.015  # 1.5% average win
        avg_loss = 0.010  # 1% average loss  
        ev = win_rate * avg_win - (1-win_rate) * avg_loss - commission
        print(f"Expected Value per trade: {ev*100:+.3f}%")

    # Save everything
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    torch.save(model.state_dict(), "models/daily_direction_v4.pth")
    
    metrics = {
        'data': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': input_features
        },
        'best_threshold': float(best_threshold),
        'accuracy': float(m['accuracy']),
        'precision': float(m['precision']),
        'recall': float(m['recall']),
        'f1': float(m['f1']),
        'win_rate': float(win_rate) if m['tp'] + m['fp'] > 0 else 0,
        'naive_baseline': float(naive_acc),
        'improvement': float(improvement),
        'confusion_matrix': {
            'tp': m['tp'], 'fp': m['fp'],
            'tn': m['tn'], 'fn': m['fn']
        }
    }
    
    with open('reports/day5_v4_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Model saved: models/daily_direction_v4.pth")
    print(f"📊 Metrics saved: reports/day5_v4_metrics.json")
    
    # Summary
    print(f"\n{'='*60}")
    print("DAY 5 SUMMARY")
    print(f"{'='*60}")
    print(f"Data: {len(X_train):,} train, {len(X_test):,} test")
    print(f"Features: {input_features} (multi-timeframe)")
    print(f"Best Accuracy: {m['accuracy']*100:.2f}%")
    print(f"Best Threshold: {best_threshold}")
    print(f"Win Rate: {win_rate*100:.1f}%" if m['tp'] + m['fp'] > 0 else "N/A")
    print(f"{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")