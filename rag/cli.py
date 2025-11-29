#!/usr/bin/env python3
"""
Scarlet Sails RAG CLI
=====================

Командная строка для извлечения данных паттернов.

Использование:
    python -m rag.cli --coin BTC --tf 1h --time "2024-11-26 14:00"
    
Сокращённая форма:
    python -m rag.cli BTC 1h "2024-11-26 14:00"
"""

import argparse
import sys
import json
from pathlib import Path

from .extractor import PatternExtractor
from .config import COINS, TIMEFRAMES, PATTERNS_DIR


def print_banner():
    """Красивый баннер."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║         SCARLET SAILS — RAG PATTERN EXTRACTOR             ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_result(data: dict):
    """Красивый вывод результата."""
    if "error" in data:
        print(f"\n❌ ОШИБКА: {data['error']}")
        return
    
    print("\n" + "="*60)
    print(f"📊 ПАТТЕРН: {data['id']}")
    print("="*60)
    
    meta = data.get('meta', {})
    print(f"\n🪙  Монета:     {meta.get('coin')}")
    print(f"⏰  Таймфрейм:  {meta.get('timeframe')}")
    print(f"📈  Тип:        {meta.get('pattern_type')}")
    print(f"↗️   Направление: {meta.get('direction')}")
    
    timing = data.get('timing', {})
    print(f"\n🕐  Время пробития: {timing.get('breakout_time_actual')}")
    print(f"🕐  Время setup:    {timing.get('setup_time')}")
    
    box = data.get('box', {})
    if box and "error" not in box:
        print(f"\n📦 BOX METRICS:")
        print(f"   Support:     {box.get('support')}")
        print(f"   Resistance:  {box.get('resistance')}")
        print(f"   Range:       {box.get('box_range_pct')}%")
        print(f"   Touches S:   {box.get('touches_support')}")
        print(f"   Touches R:   {box.get('touches_resistance')}")
        print(f"   Duration:    {box.get('duration_bars')} bars")
    
    ind = data.get('indicators_before', {})
    print(f"\n📉 ИНДИКАТОРЫ (до пробития):")
    print(f"   RSI z-score:     {ind.get('rsi_zscore')}")
    print(f"   MACD z-score:    {ind.get('macd_zscore')}")
    print(f"   ATR z-score:     {ind.get('atr_zscore')}")
    print(f"   Volume z-score:  {ind.get('volume_zscore')}")
    print(f"   Trend Up:        {ind.get('trend_up')}")
    print(f"   Vol Low:         {ind.get('vol_low')}")
    
    w = data.get('w_box', {})
    if w:
        print(f"\n🎯 W_BOX КОМПОНЕНТЫ:")
        print(f"   I_rsi:        {w.get('I_rsi')}")
        print(f"   I_volatility: {w.get('I_volatility')}")
        print(f"   I_volume:     {w.get('I_volume')}")
        print(f"   I_touches:    {w.get('I_touches')}")
        print(f"   ────────────────────")
        print(f"   W_BOX:        {w.get('W_box')} {'✅' if w.get('W_box', 0) > 0.3 else '⚠️'}")
    
    print("\n" + "="*60)


def cmd_extract(args):
    """Команда извлечения паттерна."""
    print(f"\n🔍 Поиск: {args.coin} {args.tf} @ {args.time}...")
    
    try:
        extractor = PatternExtractor(args.coin, args.tf)
        data = extractor.extract(
            breakout_time=args.time,
            pattern_type=args.type,
            direction=args.direction,
            lookback=args.lookback,
            notes=args.notes or ""
        )
        
        print_result(data)
        
        if "error" not in data:
            path = extractor.save(data)
            if path:
                print(f"\n💾 Файл: {path}")
                print(f"\n📤 Для отправки в GitHub:")
                print(f"   git add {path}")
                pattern_id = data["id"]
                print(f"   git commit -m 'Pattern: {pattern_id}'")
                print(f"   git push")
        
    except FileNotFoundError as e:
        print(f"\n❌ Файл данных не найден: {e}")
        print("   Выполни: git pull")
    except Exception as e:
        print(f"\n💥 Ошибка: {e}")
        sys.exit(1)


def cmd_list(args):
    """Команда списка паттернов."""
    patterns = list(PATTERNS_DIR.glob("*.json"))
    
    if not patterns:
        print("\n📭 Паттернов пока нет.")
        print(f"   Папка: {PATTERNS_DIR}")
        return
    
    print(f"\n📋 ПАТТЕРНЫ ({len(patterns)}):")
    print("-"*60)
    
    for p in sorted(patterns):
        with open(p, 'r') as f:
            data = json.load(f)
        
        meta = data.get('meta', {})
        w = data.get('w_box', {}).get('W_box', '?')
        print(f"   {p.stem}")
        print(f"      {meta.get('coin')} {meta.get('timeframe')} | W_box: {w}")
    
    print("-"*60)


def cmd_stats(args):
    """Команда статистики."""
    patterns = list(PATTERNS_DIR.glob("*.json"))
    
    if not patterns:
        print("\n📭 Паттернов пока нет.")
        return
    
    coins = {}
    timeframes = {}
    w_box_values = []
    
    for p in patterns:
        with open(p, 'r') as f:
            data = json.load(f)
        
        meta = data.get('meta', {})
        coin = meta.get('coin', '?')
        tf = meta.get('timeframe', '?')
        w = data.get('w_box', {}).get('W_box')
        
        coins[coin] = coins.get(coin, 0) + 1
        timeframes[tf] = timeframes.get(tf, 0) + 1
        if w is not None:
            w_box_values.append(w)
    
    print(f"\n📊 СТАТИСТИКА:")
    print(f"   Всего паттернов: {len(patterns)}")
    
    print(f"\n   По монетам:")
    for c, n in sorted(coins.items(), key=lambda x: -x[1]):
        print(f"      {c}: {n}")
    
    print(f"\n   По таймфреймам:")
    for t, n in sorted(timeframes.items()):
        print(f"      {t}: {n}")
    
    if w_box_values:
        avg_w = sum(w_box_values) / len(w_box_values)
        good = sum(1 for w in w_box_values if w > 0.3)
        print(f"\n   W_box:")
        print(f"      Средний: {avg_w:.4f}")
        print(f"      Хороших (>0.3): {good} ({100*good/len(w_box_values):.0f}%)")


def main():
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        description="Scarlet Sails RAG Pattern Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m rag.cli --coin BTC --tf 1h --time "2024-11-26 14:00"
  python -m rag.cli BTC 15m "2024-11-26 14:30" --direction short
  python -m rag.cli --list
  python -m rag.cli --stats
        """
    )
    
    # Позиционные аргументы (опциональные)
    parser.add_argument('coin', nargs='?', type=str, help='Монета (BTC, ETH, ...)')
    parser.add_argument('tf', nargs='?', type=str, choices=TIMEFRAMES, help='Таймфрейм')
    parser.add_argument('time', nargs='?', type=str, help='Время "YYYY-MM-DD HH:MM"')
    
    # Именованные аргументы
    parser.add_argument('--coin', dest='coin_named', type=str, help='Монета')
    parser.add_argument('--tf', dest='tf_named', type=str, choices=TIMEFRAMES, help='Таймфрейм')
    parser.add_argument('--time', dest='time_named', type=str, help='Время')
    
    parser.add_argument('--type', default='box_range', help='Тип паттерна (по умолчанию box_range)')
    parser.add_argument('--direction', '-d', default='long', choices=['long', 'short'], help='Направление')
    parser.add_argument('--lookback', '-l', type=int, default=48, help='Баров назад для box (по умолчанию 48)')
    parser.add_argument('--notes', '-n', type=str, help='Заметки')
    
    parser.add_argument('--list', action='store_true', help='Показать все паттерны')
    parser.add_argument('--stats', action='store_true', help='Показать статистику')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Обработка команд
    if args.list:
        cmd_list(args)
        return
    
    if args.stats:
        cmd_stats(args)
        return
    
    # Объединить позиционные и именованные
    coin = args.coin_named or args.coin
    tf = args.tf_named or args.tf
    time = args.time_named or args.time
    
    if not all([coin, tf, time]):
        parser.print_help()
        print("\n❌ Нужно указать: монету, таймфрейм и время")
        print("\nПример:")
        print('   python -m rag.cli BTC 1h "2024-11-26 14:00"')
        sys.exit(1)
    
    # Валидация
    if coin.upper() not in COINS:
        print(f"\n❌ Монета {coin} не поддерживается.")
        print(f"   Доступные: {', '.join(COINS)}")
        sys.exit(1)
    
    # Установить значения
    args.coin = coin.upper()
    args.tf = tf
    args.time = time
    
    cmd_extract(args)


if __name__ == "__main__":
    main()