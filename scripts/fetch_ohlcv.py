#!/usr/bin/env python3
"""
Загрузка OHLCV данных с Binance
"""

import os
import time
import pandas as pd
import ccxt
from datetime import datetime, timedelta

class BinanceDataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        self.data_dir = 'data/raw'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_symbol_data(self, symbol, timeframe, days=730):
        """Загрузка данных для одного символа"""
        print(f"Загрузка {symbol} {timeframe}...")
        
        # Вычисление диапазона дат
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        since = int(start_date.timestamp() * 1000)
        limit = 1000
        all_data = []
        
        while since < int(end_date.timestamp() * 1000):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                print(f"  Загружено: {len(all_data)} свечей")
                time.sleep(0.1)  # Ограничение запросов
                
            except Exception as e:
                print(f"Ошибка: {e}")
                time.sleep(5)
                continue
        
        # Конвертация в DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Удаление дубликатов
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        
        # Сохранение
        filename = f"{symbol.replace('/', '_')}_{timeframe}.parquet"
        filepath = os.path.join(self.data_dir, filename)
        df.to_parquet(filepath)
        
        print(f"  Сохранено: {len(df)} свечей в {filename}")
        return len(df)

def main():
    fetcher = BinanceDataFetcher()
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    timeframes = ['1m', '15m', '1h']
    
    total_downloaded = 0
    
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                count = fetcher.fetch_symbol_data(symbol, timeframe)
                total_downloaded += count
            except Exception as e:
                print(f"ОШИБКА {symbol} {timeframe}: {e}")
                continue
    
    print(f"\nЗагрузка завершена. Всего свечей: {total_downloaded}")
    
    # Проверка размеров файлов
    print("\nРазмеры файлов:")
    for filename in os.listdir(fetcher.data_dir):
        if filename.endswith('.parquet'):
            filepath = os.path.join(fetcher.data_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
