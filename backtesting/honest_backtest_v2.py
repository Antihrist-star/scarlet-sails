"""
Honest Backtest V2 - Fixed Horizon + Triple Barrier + Hysteresis
================================================================================
Fixes from V1:
1. Fixed hold period (matches model horizon - 3 days)
2. Triple barrier exit (TP/SL/Time)
3. Hysteresis (cooldown between trades)
4. All metrics (PF, MDD, Sharpe, Calmar)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt


@dataclass
class Trade:
    """Single trade record"""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'tp', 'sl', 'time'


class HonestBacktestV2:
    """
    Honest Backtest Engine V2
    
    Key improvements:
    - Fixed hold period (3 days = 288 bars for 15min)
    - Triple barrier: TP/SL/Time
    - Hysteresis: cooldown between trades
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,      # 0.1% per side
        slippage: float = 0.0005,       # 0.05% slippage
        position_size_pct: float = 0.95, # 95% of capital
        take_profit: float = 0.02,      # 2% TP
        stop_loss: float = 0.01,        # 1% SL
        max_hold_bars: int = 288,       # 3 days for 15min bars
        cooldown_bars: int = 10         # 10 bars between trades
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size_pct = position_size_pct
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.max_hold_bars = max_hold_bars
        self.cooldown_bars = cooldown_bars
        
        self.trades: List[Trade] = []
        self.equity_curve = []
        
    def run(
        self,
        ohlcv: pd.DataFrame,
        signals: np.ndarray
    ) -> dict:
        """
        Run backtest with triple barrier logic
        
        Args:
            ohlcv: DataFrame with ['open', 'high', 'low', 'close', 'volume']
            signals: Binary array (0=no trade, 1=buy signal)
        
        Returns:
            dict with all metrics
        """
        capital = self.initial_capital
        position = None  # None or dict with position info
        last_exit_bar = -self.cooldown_bars  # Allow first trade immediately
        
        self.equity_curve = [capital]

        for i in range(len(signals)):
            current_bar = ohlcv.iloc[i]

            # Обновляем кривую капитала (даже если нет позиции)
            current_equity = capital
            if position is not None:
                # Добавляем стоимость открытой позиции (Mark-to-Market)
                current_equity += position['shares'] * current_bar['close']
            self.equity_curve.append(current_equity)

            # Проверяем условия выхода, если в позиции
            if position is not None:
                bars_held = i - position['entry_bar']
                current_price = current_bar['close']

                # Рассчитываем P&L в процентах от цены входа
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']

                # Определяем, нужно ли выходить
                exit_now = False
                exit_reason = None

                # 1. Take Profit: проверяем high (если цена поднялась выше TP)
                if pnl_pct >= self.take_profit:
                    # Выход по цене, ближайшей к TP (high могла превысить TP)
                    exit_price = max(current_bar['open'], current_bar['high'], position['entry_price'] * (1 + self.take_profit))
                    exit_reason = 'tp'
                    exit_now = True
                # 2. Stop Loss: проверяем low (если цена упала ниже SL)
                elif pnl_pct <= -self.stop_loss:
                    # Выход по цене, ближайшей к SL (low могла упасть ниже SL)
                    exit_price = min(current_bar['open'], current_bar['low'], position['entry_price'] * (1 - self.stop_loss))
                    exit_reason = 'sl'
                    exit_now = True
                # 3. Max Hold Time
                elif bars_held >= self.max_hold_bars:
                    exit_price = current_bar['close'] # Выход по цене закрытия
                    exit_reason = 'time'
                    exit_now = True

                if exit_now:
                    # Применяем проскальзывание и комиссию к цене выхода
                    exit_price_with_costs = exit_price * (1 - self.slippage) * (1 - self.commission)

                    # Рассчитываем выручку от продажи
                    proceeds = position['shares'] * exit_price_with_costs

                    # Рассчитываем P&L
                    pnl = proceeds - position['initial_investment'] # Используем изначальную инвестированную сумму
                    pnl_pct_final = pnl / position['initial_investment']

                    # Записываем сделку
                    trade = Trade(
                        entry_bar=position['entry_bar'],
                        exit_bar=i,
                        entry_price=position['entry_price'],
                        exit_price=exit_price_with_costs, # Цена выхода после всех издержек
                        pnl=pnl,
                        pnl_pct=pnl_pct_final,
                        exit_reason=exit_reason
                    )
                    self.trades.append(trade)

                    # --- ИСПРАВЛЕНАЯ ЛОГИКА ОБНОВЛЕНИЯ КАПИТАЛА ---
                    # Капитал увеличивается на *всю* вырученную сумму от продажи
                    capital = capital + proceeds

                    # Сбрасываем позицию
                    position = None
                    last_exit_bar = i

            # Проверяем условия входа, если нет позиции и прошёл cooldown
            elif signals[i] == 1 and (i - last_exit_bar) >= self.cooldown_bars:
                # Цена входа с учётом проскальзывания
                entry_price_with_slippage = current_bar['open'] * (1 + self.slippage)

                # Рассчитываем размер позиции
                # Допустим, используем фиксированный процент капитала
                available_capital_for_trade = capital * self.position_size_pct
                shares_to_buy = available_capital_for_trade / entry_price_with_slippage

                # Рассчитываем стоимость покупки (включая комиссию)
                cost_of_shares = shares_to_buy * entry_price_with_slippage
                total_cost_with_commission = cost_of_shares * (1 + self.commission)

                # Проверяем, хватает ли капитала
                if total_cost_with_commission <= capital:
                    # Создаём позицию
                    position = {
                        'entry_bar': i,
                        'entry_price': entry_price_with_slippage,
                        'shares': shares_to_buy,
                        'initial_investment': cost_of_shares # Сохраняем *чистую* стоимость без комиссии за покупку
                    }
                    # --- ИСПРАВЛЕНАЯ ЛОГИКА ОБНОВЛЕНИЯ КАПИТАЛА ---
                    # Капитал уменьшается на *всю* сумму покупки (включая комиссию)
                    capital = capital - total_cost_with_commission

        # Close any remaining position at end
        if position is not None:
            final_bar = ohlcv.iloc[-1]
            exit_price = final_bar['close'] * (1 - self.slippage)  # Sell with slippage
            proceeds = position['shares'] * exit_price * (1 - self.commission)
            pnl = proceeds - position['initial_investment']
            pnl_pct = pnl / position['initial_investment']

            trade = Trade(
                entry_bar=position['entry_bar'],
                exit_bar=len(ohlcv) - 1,
                entry_price=position['entry_price'],
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason='forced_close'
            )
            self.trades.append(trade)
            capital += proceeds # Обновляем капитал после forced close
            self.equity_curve[-1] = capital # Обновляем последнее значение кривой

        # Calculate all metrics
        return self._calculate_metrics()
    
    def _calculate_metrics(self) -> dict:
        """Calculate all performance metrics"""
        
        if len(self.trades) == 0:
            return {
                'total_return_pct': 0.0,
                'total_pnl': 0.0,
                'n_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'avg_bars_held': 0.0,
                'exit_reasons': {}
            }
        
        # Basic stats
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]
        
        n_trades = len(self.trades)
        n_wins = len(wins)
        n_losses = len(losses)
        
        win_rate = n_wins / n_trades if n_trades > 0 else 0
        
        # Profit Factor
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 1e-10
        profit_factor = total_wins / total_losses
        
        # Total Return
        final_capital = self.equity_curve[-1]
        total_pnl = final_capital - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Max Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown_pct = abs(drawdown.min())
        
        # Sharpe Ratio
        returns = np.diff(equity_array) / equity_array[:-1]
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 96)  # Annualized for 15min bars
        else:
            sharpe_ratio = 0.0
        
        # Calmar Ratio
        annualized_return = total_return_pct  # Simplified
        calmar_ratio = (annualized_return / max_drawdown_pct) if max_drawdown_pct > 0 else 0
        
        # Averages
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_bars_held = np.mean([t.exit_bar - t.entry_bar for t in self.trades])
        
        # Exit reasons
        exit_reasons = {}
        for t in self.trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        
        return {
            'total_return_pct': total_return_pct,
            'total_pnl': total_pnl,
            'n_trades': n_trades,
            'n_wins': n_wins,
            'n_losses': n_losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_bars_held': avg_bars_held,
            'exit_reasons': exit_reasons
        }
    
    def print_report(self):
        """Print detailed backtest report"""
        metrics = self._calculate_metrics()
        
        print("\n" + "="*80)
        print("HONEST BACKTEST V2 - RESULTS")
        print("="*80)
        print(f"Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"Final Capital:          ${self.equity_curve[-1]:,.2f}")
        print(f"Total P&L:              ${metrics['total_pnl']:,.2f}")
        print(f"Total Return:           {metrics['total_return_pct']:.2f}%")
        print("-"*80)
        print(f"Number of Trades:       {metrics['n_trades']}")
        print(f"Wins:                   {metrics['n_wins']}")
        print(f"Losses:                 {metrics['n_losses']}")
        print(f"Win Rate:               {metrics['win_rate']:.2%}")
        print(f"Avg Win:                ${metrics['avg_win']:.2f}")
        print(f"Avg Loss:               ${metrics['avg_loss']:.2f}")
        print(f"Avg Bars Held:          {metrics['avg_bars_held']:.1f}")
        print("-"*80)
        print(f"Profit Factor:          {metrics['profit_factor']:.4f}")
        print(f"Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:           {metrics['sharpe_ratio']:.4f}")
        print(f"Calmar Ratio:           {metrics['calmar_ratio']:.4f}")
        print("-"*80)
        print("Exit Reasons:")
        for reason, count in metrics['exit_reasons'].items():
            pct = (count / metrics['n_trades']) * 100
            print(f"  {reason:12s}: {count:4d} ({pct:5.1f}%)")
        print("="*80)
        
        # Goal check
        print("\n🎯 WEEK 3 GOALS CHECK:")
        pf_status = "✅" if metrics['profit_factor'] >= 2.0 else "❌"
        dd_status = "✅" if metrics['max_drawdown_pct'] <= 15.0 else "❌"
        
        print(f"{pf_status} Profit Factor ≥ 2.0: {metrics['profit_factor']:.4f}")
        print(f"{dd_status} Max Drawdown ≤ 15%: {metrics['max_drawdown_pct']:.2f}%")
        
        if metrics['profit_factor'] >= 2.0 and metrics['max_drawdown_pct'] <= 15.0:
            print("\n🎉 GOALS ACHIEVED!")
        elif metrics['profit_factor'] >= 1.5:
            print("\n⚠️  Promising! Needs optimization.")
        else:
            print("\n❌ Not profitable yet. Needs major improvements.")
    
    def plot_results(self, save_path='reports/backtest_v2_results.png'):
        """Plot equity curve and drawdown"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curve
        axes[0].plot(self.equity_curve, label='Portfolio Value', linewidth=2)
        axes[0].axhline(self.initial_capital, color='gray', linestyle='--', 
                       alpha=0.5, label='Initial Capital')
        axes[0].set_title('Equity Curve (V2 - Triple Barrier)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Bar Number')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, 
                             alpha=0.3, color='red', label='Drawdown')
        axes[1].plot(drawdown, color='darkred', linewidth=1.5)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Bar Number')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved: {save_path}")
        plt.close()
