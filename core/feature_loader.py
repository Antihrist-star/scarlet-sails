"""
Feature Loader for Scarlet Sails Phase 3
=========================================
Loads 75-column feature files from data/features/*.parquet

Feature Structure (75 columns):
- OHLCV (5): open, high, low, close, volume
- Normalized (variable): norm_*_zscore, norm_*_pctile
- Derivatives (variable): deriv_close_diff1, deriv_close_diff2, deriv_close_roc5, ...
- Regime (variable): regime_*
- Cross (variable): cross_*
- Divergence (variable): div_*
- Time (variable): time_*

Author: STAR_ANT + Claude
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature loading."""
    data_dir: str = "data/features"
    required_columns: List[str] = None
    
    def __post_init__(self):
        if self.required_columns is None:
            # Минимальные колонки для работы стратегий
            self.required_columns = ['open', 'high', 'low', 'close', 'volume']


class FeatureLoader:
    """
    Loader for 75-column feature parquet files.
    
    Supports:
    - Single asset loading
    - Multi-asset batch loading
    - Date range filtering
    - Feature validation
    - Column subset selection
    """
    
    SUPPORTED_COINS = [
        'ALGO', 'AVAX', 'BTC', 'DOT', 'ENA', 'ETH', 
        'HBAR', 'LDO', 'LINK', 'LTC', 'ONDO', 'SOL', 'SUI', 'UNI'
    ]
    
    SUPPORTED_TIMEFRAMES = ['15m', '1h', '4h', '1d', '1m']
    
    def __init__(self, data_dir: str = "data/features"):
        """
        Initialize FeatureLoader.
        
        Parameters
        ----------
        data_dir : str
            Path to features directory (default: data/features)
        """
        self.data_dir = Path(data_dir)
        self._feature_info: Dict[str, Dict] = {}
    
    def get_file_path(self, coin: str, timeframe: str) -> Path:
        """Get path to feature file."""
        filename = f"{coin}_USDT_{timeframe}_features.parquet"
        return self.data_dir / filename
    
    def file_exists(self, coin: str, timeframe: str) -> bool:
        """Check if feature file exists."""
        return self.get_file_path(coin, timeframe).exists()
    
    def load_features(
        self,
        coin: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load features for a single asset.
        
        Parameters
        ----------
        coin : str
            Coin symbol (e.g., 'BTC', 'ETH', 'ENA')
        timeframe : str
            Timeframe (e.g., '15m', '1h', '4h', '1d')
        start_date : str, optional
            Start date filter (YYYY-MM-DD)
        end_date : str, optional
            End date filter (YYYY-MM-DD)
        columns : list, optional
            Subset of columns to load (None = all)
        validate : bool
            Whether to validate data quality
            
        Returns
        -------
        pd.DataFrame
            Feature dataframe with datetime index
        """
        coin = coin.upper()
        
        # Validate inputs
        if coin not in self.SUPPORTED_COINS:
            raise ValueError(f"Unsupported coin: {coin}. Supported: {self.SUPPORTED_COINS}")
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {self.SUPPORTED_TIMEFRAMES}")
        
        file_path = self.get_file_path(coin, timeframe)
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")
        
        # Load parquet
        if columns:
            df = pd.read_parquet(file_path, columns=columns)
        else:
            df = pd.read_parquet(file_path)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            elif 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
        
        # Filter by date range
        if start_date:
            start_ts = pd.Timestamp(start_date)
            if df.index.tz is not None and start_ts.tz is None:
                start_ts = start_ts.tz_localize(df.index.tz)
            df = df[df.index >= start_ts]
        
        if end_date:
            end_ts = pd.Timestamp(end_date)
            if df.index.tz is not None and end_ts.tz is None:
                end_ts = end_ts.tz_localize(df.index.tz)
            df = df[df.index <= end_ts]
        
        # Store feature info
        self._feature_info[f"{coin}_{timeframe}"] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'date_range': (df.index[0], df.index[-1]),
            'nan_count': df.isna().sum().sum()
        }
        
        # Validate if requested
        if validate:
            self._validate_features(df, coin, timeframe)
        
        return df
    
    def load_multiple(
        self,
        coins: List[str],
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load features for multiple coins.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping coin -> DataFrame
        """
        result = {}
        errors = []
        
        for coin in coins:
            try:
                df = self.load_features(
                    coin=coin,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    columns=columns
                )
                result[coin] = df
            except Exception as e:
                errors.append(f"{coin}: {str(e)}")
        
        if errors:
            print(f"Warning: Failed to load {len(errors)} coins:")
            for err in errors:
                print(f"  - {err}")
        
        return result
    
    def _validate_features(self, df: pd.DataFrame, coin: str, timeframe: str) -> None:
        """Validate feature data quality."""
        issues = []
        
        # Check for required OHLCV columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            issues.append(f"Missing required columns: {missing}")
        
        # Check for NaN in OHLCV
        if 'close' in df.columns:
            nan_pct = df['close'].isna().mean() * 100
            if nan_pct > 1:
                issues.append(f"High NaN rate in 'close': {nan_pct:.2f}%")
        
        # Check for negative prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values in '{col}'")
        
        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high = (df['high'] < df['low']).sum()
            if invalid_high > 0:
                issues.append(f"high < low in {invalid_high} rows")
        
        if issues:
            print(f"⚠️ Validation warnings for {coin}_{timeframe}:")
            for issue in issues:
                print(f"   - {issue}")
    
    def get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Group columns by feature type.
        
        Returns
        -------
        Dict mapping feature group -> list of columns
        """
        groups = {
            'ohlcv': [],
            'normalized': [],
            'derivatives': [],
            'regime': [],
            'cross': [],
            'divergence': [],
            'time': [],
            'other': []
        }
        
        for col in df.columns:
            if col in ['open', 'high', 'low', 'close', 'volume']:
                groups['ohlcv'].append(col)
            elif col.startswith('norm_'):
                groups['normalized'].append(col)
            elif col.startswith('deriv_'):
                groups['derivatives'].append(col)
            elif col.startswith('regime_'):
                groups['regime'].append(col)
            elif col.startswith('cross_'):
                groups['cross'].append(col)
            elif col.startswith('div_'):
                groups['divergence'].append(col)
            elif col.startswith('time_'):
                groups['time'].append(col)
            else:
                groups['other'].append(col)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups
    
    def get_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only ML-relevant features (exclude OHLCV).
        
        Returns
        -------
        pd.DataFrame with only norm_*, deriv_*, regime_*, etc.
        """
        exclude = ['open', 'high', 'low', 'close', 'volume']
        ml_cols = [col for col in df.columns if col not in exclude]
        return df[ml_cols]
    
    def describe_features(self, coin: str, timeframe: str) -> None:
        """Print feature summary for a dataset."""
        df = self.load_features(coin, timeframe, validate=False)
        groups = self.get_feature_groups(df)
        
        print(f"\n{'='*60}")
        print(f"FEATURE SUMMARY: {coin}_{timeframe}")
        print(f"{'='*60}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Total rows: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"NaN total: {df.isna().sum().sum()}")
        print(f"\nFeature groups:")
        for group, cols in groups.items():
            print(f"  {group}: {len(cols)} columns")
            if len(cols) <= 5:
                print(f"    → {cols}")
            else:
                print(f"    → {cols[:3]} ... {cols[-2:]}")
        print(f"{'='*60}\n")


def load_features_for_backtest(
    coin: str,
    timeframe: str,
    data_dir: str = "data/features",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to load features for backtesting.
    
    This is the main entry point for Phase 3 integration.
    """
    loader = FeatureLoader(data_dir=data_dir)
    return loader.load_features(
        coin=coin,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )


# Quick test
if __name__ == "__main__":
    import sys
    
    # Default test
    coin = sys.argv[1] if len(sys.argv) > 1 else "ENA"
    timeframe = sys.argv[2] if len(sys.argv) > 2 else "15m"
    
    loader = FeatureLoader()
    
    # Check if file exists first
    if loader.file_exists(coin, timeframe):
        loader.describe_features(coin, timeframe)
    else:
        print(f"File not found: {loader.get_file_path(coin, timeframe)}")
        print(f"Available coins: {loader.SUPPORTED_COINS}")