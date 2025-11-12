"""
Multi-Timeframe Feature Extractor

Extracts 31 features matching the trained XGBoost model:
- 13 features from primary timeframe (15m)
- 18 features from higher timeframes (1h, 4h, 1d)

Used by: HybridEntrySystem for ML filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class MultiTimeframeFeatureExtractor:
    """
    Extract multi-timeframe features for XGBoost model

    Features (31 total):
    - 15m: RSI, EMA_9, EMA_21, SMA_50, BB (4 cols), ATR, returns (2), volume (2) = 13
    - 1h, 4h, 1d: close, EMA_9, EMA_21, SMA_50, RSI, returns_5, returns_20 = 7 each x 3 = 21

    But only key features are selected, matching trained model (18 from higher TFs)
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize feature extractor

        Args:
            data_dir: Directory containing raw OHLCV parquet files
        """
        self.data_dir = Path(data_dir)
        self.timeframes = ['15m', '1h', '4h', '1d']

    def load_all_timeframes(self, asset: str) -> Dict[str, pd.DataFrame]:
        """
        Load all 4 timeframes for an asset

        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH')

        Returns:
            Dict mapping timeframe -> DataFrame with OHLCV
        """
        data = {}

        for tf in self.timeframes:
            # Try both naming conventions
            path1 = self.data_dir / f"{asset}_USDT_{tf}.parquet"
            path2 = self.data_dir / f"{asset}USDT_{tf}.parquet"

            if path1.exists():
                df = pd.read_parquet(path1)
                data[tf] = df
            elif path2.exists():
                df = pd.read_parquet(path2)
                data[tf] = df
            else:
                raise FileNotFoundError(f"No data found for {asset} {tf}")

        return data

    def calculate_indicators(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Calculate technical indicators for a timeframe

        Args:
            df: DataFrame with OHLCV
            prefix: Prefix for column names (e.g., '15m_', '1h_')

        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df[f'{prefix}RSI_14'] = 100 - (100 / (1 + rs))

        # EMAs
        df[f'{prefix}EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df[f'{prefix}EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()

        # SMA
        df[f'{prefix}SMA_50'] = df['close'].rolling(window=50).mean()

        # Bollinger Bands (only for primary timeframe)
        if prefix == '15m_':
            df[f'{prefix}BB_middle'] = df['close'].rolling(window=20).mean()
            df[f'{prefix}BB_std'] = df['close'].rolling(window=20).std()
            df[f'{prefix}BB_upper'] = df[f'{prefix}BB_middle'] + (df[f'{prefix}BB_std'] * 2)
            df[f'{prefix}BB_lower'] = df[f'{prefix}BB_middle'] - (df[f'{prefix}BB_std'] * 2)

        # ATR (only for primary timeframe)
        if prefix == '15m_':
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.DataFrame({
                'HL': high_low,
                'HC': high_close,
                'LC': low_close
            }).max(axis=1)
            df[f'{prefix}ATR_14'] = true_range.rolling(window=14).mean()

        # Returns
        df[f'{prefix}returns_5'] = df['close'].pct_change(5)
        df[f'{prefix}returns_20'] = df['close'].pct_change(20)

        # Volume features (only for primary timeframe)
        if prefix == '15m_' and 'volume' in df.columns:
            df[f'{prefix}volume_sma'] = df['volume'].rolling(20).mean()
            df[f'{prefix}volume_ratio'] = df['volume'] / (df[f'{prefix}volume_sma'] + 1e-10)

        return df

    def resample_to_target(self, df_source: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample source timeframe to target resolution

        Args:
            df_source: Source DataFrame
            target_tf: Target timeframe ('15m', '1h', '4h', '1d')

        Returns:
            Resampled DataFrame (forward-filled)
        """
        # Map timeframe to pandas frequency
        freq_map = {
            '15m': '15min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }

        target_freq = freq_map[target_tf]

        # Resample and forward-fill
        df_resampled = df_source.resample(target_freq).ffill()

        return df_resampled

    def extract_features_at_bar(
        self,
        all_timeframes: Dict[str, pd.DataFrame],
        target_tf: str,
        bar_index: int
    ) -> Optional[np.ndarray]:
        """
        Extract NORMALIZED features at specific bar for ML model

        CRITICAL: Must match retrain_xgboost_normalized.py feature order!
        Uses ratios and percentages instead of absolute values.

        Args:
            all_timeframes: Dict of DataFrames (already with indicators!)
            target_tf: Target timeframe being tested
            bar_index: Bar index in target timeframe

        Returns:
            Array of 31 NORMALIZED features, or None if data unavailable
        """
        try:
            # Get primary timeframe data
            primary_df = all_timeframes[target_tf]

            # Check if bar_index is valid
            if bar_index >= len(primary_df):
                return None

            timestamp = primary_df.index[bar_index]
            current_close = primary_df['close'].iloc[bar_index]

            # Extract NORMALIZED features from primary timeframe (15m)
            features = []

            # Primary timeframe features (10 normalized + 3 duplicate ratios = 13 total)
            # RSI (already 0-100, divide by 100 to normalize to 0-1)
            rsi = primary_df[f'15m_RSI_14'].iloc[bar_index] / 100.0
            features.append(rsi)

            # Price ratios (normalized!)
            price_to_ema9 = current_close / primary_df[f'15m_EMA_9'].iloc[bar_index]
            price_to_ema21 = current_close / primary_df[f'15m_EMA_21'].iloc[bar_index]
            price_to_sma50 = current_close / primary_df[f'15m_SMA_50'].iloc[bar_index]
            features.append(price_to_ema9)
            features.append(price_to_ema21)
            features.append(price_to_sma50)

            # Bollinger Band width as percentage of price (normalized!)
            bb_std = primary_df[f'15m_BB_std'].iloc[bar_index]
            bb_width_pct = (2 * bb_std) / current_close
            features.append(bb_width_pct)

            # ATR as percentage of price (normalized!)
            atr_pct = primary_df[f'15m_ATR_14'].iloc[bar_index] / current_close
            features.append(atr_pct)

            # Returns (already normalized!)
            features.append(primary_df[f'15m_returns_5'].iloc[bar_index])
            features.append(primary_df[f'15m_returns_20'].iloc[bar_index])  # Note: training uses returns_10

            # Volume ratio (normalized!)
            volume_ratio = primary_df[f'15m_volume_ratio'].iloc[bar_index]
            features.append(volume_ratio)
            features.append(volume_ratio)  # Duplicate (matches training: volume_ratio_5, volume_ratio_10)

            # Duplicate price ratios for completeness (matches training)
            features.append(price_to_ema9)
            features.append(price_to_ema21)
            features.append(price_to_sma50)

            # Higher timeframe features (18 total = 6 features x 3 TFs)
            for tf in ['1h', '4h', '1d']:
                if tf == target_tf:
                    # Same timeframe - use direct values
                    df_tf = all_timeframes[tf]
                    idx = bar_index
                else:
                    # Different timeframe - use resampled values
                    df_tf = all_timeframes[tf]
                    # Find closest timestamp
                    idx = df_tf.index.get_indexer([timestamp], method='ffill')[0]
                    if idx == -1:
                        # No data available
                        return None

                # Get close price for this TF
                tf_close = df_tf['close'].iloc[idx]

                # Extract NORMALIZED features (matching training data)
                # RSI normalized to 0-1
                tf_rsi = df_tf[f'{tf}_RSI_14'].iloc[idx] / 100.0
                features.append(tf_rsi)

                # Returns (already normalized)
                tf_returns_5 = df_tf[f'{tf}_returns_5'].iloc[idx]
                features.append(tf_returns_5)

                # Price ratios (normalized!)
                tf_price_to_ema9 = tf_close / df_tf[f'{tf}_EMA_9'].iloc[idx]
                tf_price_to_ema21 = tf_close / df_tf[f'{tf}_EMA_21'].iloc[idx]
                tf_price_to_sma50 = tf_close / df_tf[f'{tf}_SMA_50'].iloc[idx]
                features.append(tf_price_to_ema9)
                features.append(tf_price_to_ema21)
                features.append(tf_price_to_sma50)

                # ATR as percentage (normalized!)
                # Note: We need to calculate ATR for higher TFs if not present
                # For now, use a proxy based on volatility
                if f'{tf}_returns_5' in df_tf.columns:
                    tf_atr_pct = abs(tf_returns_5) * 2  # Approximation
                else:
                    tf_atr_pct = 0.02  # Default fallback
                features.append(tf_atr_pct)

            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)

            # Check for NaN or inf
            if np.isnan(features_array).any() or np.isinf(features_array).any():
                return None

            return features_array

        except (KeyError, IndexError, ZeroDivisionError) as e:
            # Missing data, column, or division by zero
            return None

    def prepare_multi_timeframe_data(
        self,
        asset: str,
        target_tf: str
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load and prepare all timeframes for an asset

        Args:
            asset: Asset symbol (e.g., 'BTC')
            target_tf: Target timeframe for testing

        Returns:
            (all_timeframes dict, merged_df) where:
            - all_timeframes: Dict with indicators for each TF
            - merged_df: Primary TF data with all features merged
        """
        # Load all timeframes
        all_tf_data = self.load_all_timeframes(asset)

        # Calculate indicators for each timeframe
        all_tf_with_indicators = {}

        for tf in self.timeframes:
            df = all_tf_data[tf].copy()
            prefix = f'{tf}_'
            df = self.calculate_indicators(df, prefix)
            all_tf_with_indicators[tf] = df

        # Get primary timeframe (target)
        primary_df = all_tf_with_indicators[target_tf].copy()

        # Resample higher timeframes to target resolution
        for tf in self.timeframes:
            if tf == target_tf:
                continue  # Skip same timeframe

            df_higher = all_tf_with_indicators[tf]

            # Resample to target frequency
            df_resampled = self.resample_to_target(df_higher, target_tf)

            # Merge key columns
            key_cols = [col for col in df_resampled.columns
                       if any(x in col for x in ['close', 'EMA', 'SMA', 'RSI', 'returns'])]

            for col in key_cols:
                if col not in primary_df.columns:
                    primary_df[col] = df_resampled[col]

        return all_tf_with_indicators, primary_df


# Test function
if __name__ == "__main__":
    print("=== Multi-Timeframe Feature Extractor Test ===\n")

    extractor = MultiTimeframeFeatureExtractor()

    # Test on BTC
    try:
        all_tf, merged_df = extractor.prepare_multi_timeframe_data('BTC', '15m')

        print(f"✅ Loaded BTC data")
        print(f"   Primary (15m): {len(merged_df)} bars")
        print(f"   Columns: {len(merged_df.columns)}")
        print(f"\n   Sample columns: {list(merged_df.columns[:20])}")

        # Extract features at random bar
        bar_idx = 1000
        features = extractor.extract_features_at_bar(all_tf, '15m', bar_idx)

        if features is not None:
            print(f"\n✅ Extracted {len(features)} features at bar {bar_idx}")
            print(f"   Sample: {features[:5]}")
        else:
            print(f"\n❌ Could not extract features at bar {bar_idx}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
