#!/usr/bin/env python3
"""
Тест PyTorch и CUDA
"""

import torch
import torch.nn as nn
import sys

def test_cuda():
    print("=== PyTorch CUDA Test ===")
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступен: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"Количество GPU: {torch.cuda.device_count()}")
        print(f"Текущий GPU: {torch.cuda.current_device()}")
        print(f"Имя GPU: {torch.cuda.get_device_name()}")
        
        # Тест вычислений на GPU
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"Матричное умножение 1000x1000 на GPU: {elapsed_time:.2f} мс")
        
        # Тест простой нейросети
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        net = TestNet().to(device)
        test_input = torch.randn(32, 10).to(device)
        output = net(test_input)
        
        print(f"Нейросеть тест - входы: {test_input.shape}, выходы: {output.shape}")
        print("GPU работает корректно!")
        
    else:
        print("CUDA недоступен - будет использован CPU")
        
        # CPU тест
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        z = torch.matmul(x, y)
        print("CPU тест пройден")

def test_dependencies():
    print("\n=== Тест зависимостей ===")
    dependencies = [
        'pandas', 'numpy', 'ccxt', 
        'sklearn', 'joblib', 'matplotlib'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - НЕ УСТАНОВЛЕН")

if __name__ == "__main__":
    test_cuda()
    test_dependencies()
    print("\nТест завершен.")
