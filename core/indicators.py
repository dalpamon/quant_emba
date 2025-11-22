"""
Technical Indicators Module for Quantitative Trading

Provides comprehensive technical indicators for feature engineering and strategy development.
Optimized for Korean stock market (KRX/KOSPI/KOSDAQ) trading.

Author: Agentic Backtesting Framework
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.

    Supports:
    - Momentum indicators (RSI, MACD, Stochastic, Williams %R)
    - Volatility indicators (Bollinger Bands, ATR, Keltner Channels)
    - Trend indicators (SMA, EMA, ADX, Parabolic SAR)
    - Volume indicators (OBV, MFI, VWAP, Accumulation/Distribution)
    - Statistical measures (Geometric/Arithmetic returns, Z-scores)
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data.

        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                  Can be single stock or MultiIndex (ticker, date)
        """
        self.data = data.copy()
        self.is_multiindex = isinstance(data.index, pd.MultiIndex)

    # ==================== MOMENTUM INDICATORS ====================

    def rsi(self, period: int = 14, column: str = 'close') -> pd.Series:
        """
        Relative Strength Index (RSI)

        Measures magnitude of recent price changes to evaluate overbought/oversold conditions.
        Range: 0-100 (>70 overbought, <30 oversold)

        Args:
            period: Lookback period (default 14)
            column: Price column to use

        Returns:
            RSI values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].apply(
                lambda x: self._calculate_rsi(x, period)
            )
        else:
            return self._calculate_rsi(self.data[column], period)

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI for a single price series."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD)

        Trend-following momentum indicator showing relationship between two EMAs.

        Args:
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            column: Price column to use

        Returns:
            DataFrame with columns: ['macd', 'signal', 'histogram']
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].apply(
                lambda x: self._calculate_macd(x, fast_period, slow_period, signal_period)
            )
        else:
            return self._calculate_macd(self.data[column], fast_period, slow_period, signal_period)

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> pd.DataFrame:
        """Calculate MACD for a single price series."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line

        result = pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        })
        return result

    def stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator (%K and %D)

        Compares closing price to price range over time.
        Range: 0-100 (>80 overbought, <20 oversold)

        Args:
            k_period: %K lookback period (default 14)
            d_period: %D smoothing period (default 3)
            smooth_k: %K smoothing period (default 3)

        Returns:
            DataFrame with columns: ['%K', '%D']
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_stochastic(x, k_period, d_period, smooth_k)
            )
        else:
            return self._calculate_stochastic(self.data, k_period, d_period, smooth_k)

    def _calculate_stochastic(
        self,
        data: pd.DataFrame,
        k_period: int,
        d_period: int,
        smooth_k: int
    ) -> pd.DataFrame:
        """Calculate Stochastic for a single stock."""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()

        k_fast = 100 * (data['close'] - low_min) / (high_max - low_min)
        k_slow = k_fast.rolling(window=smooth_k).mean()  # %K
        d = k_slow.rolling(window=d_period).mean()  # %D

        result = pd.DataFrame({
            '%K': k_slow,
            '%D': d
        })
        return result

    def williams_r(self, period: int = 14) -> pd.Series:
        """
        Williams %R

        Momentum indicator measuring overbought/oversold levels.
        Range: -100 to 0 (>-20 overbought, <-80 oversold)

        Args:
            period: Lookback period (default 14)

        Returns:
            Williams %R values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_williams_r(x, period)
            )
        else:
            return self._calculate_williams_r(self.data, period)

    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R for a single stock."""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()

        wr = -100 * (high_max - data['close']) / (high_max - low_min)
        return wr

    # ==================== VOLATILITY INDICATORS ====================

    def bollinger_bands(
        self,
        period: int = 20,
        num_std: float = 2.0,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Bollinger Bands

        Volatility bands placed above and below moving average.

        Args:
            period: MA period (default 20)
            num_std: Number of standard deviations (default 2.0)
            column: Price column to use

        Returns:
            DataFrame with columns: ['upper', 'middle', 'lower', 'bandwidth', 'pct_b']
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].apply(
                lambda x: self._calculate_bollinger_bands(x, period, num_std)
            )
        else:
            return self._calculate_bollinger_bands(self.data[column], period, num_std)

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int,
        num_std: float
    ) -> pd.DataFrame:
        """Calculate Bollinger Bands for a single price series."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        # Additional metrics
        bandwidth = (upper - lower) / middle  # Bandwidth as % of middle band
        pct_b = (prices - lower) / (upper - lower)  # %B position within bands

        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'bandwidth': bandwidth,
            'pct_b': pct_b
        })
        return result

    def atr(self, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)

        Measures market volatility by decomposing the entire range of price movement.

        Args:
            period: Lookback period (default 14)

        Returns:
            ATR values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_atr(x, period)
            )
        else:
            return self._calculate_atr(self.data, period)

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR for a single stock."""
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def keltner_channels(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        atr_multiplier: float = 2.0
    ) -> pd.DataFrame:
        """
        Keltner Channels

        Volatility-based envelopes set above/below EMA using ATR.

        Args:
            ema_period: EMA period (default 20)
            atr_period: ATR period (default 10)
            atr_multiplier: ATR multiplier (default 2.0)

        Returns:
            DataFrame with columns: ['upper', 'middle', 'lower']
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_keltner_channels(
                    x, ema_period, atr_period, atr_multiplier
                )
            )
        else:
            return self._calculate_keltner_channels(
                self.data, ema_period, atr_period, atr_multiplier
            )

    def _calculate_keltner_channels(
        self,
        data: pd.DataFrame,
        ema_period: int,
        atr_period: int,
        atr_multiplier: float
    ) -> pd.DataFrame:
        """Calculate Keltner Channels for a single stock."""
        middle = data['close'].ewm(span=ema_period, adjust=False).mean()
        atr = self._calculate_atr(data, atr_period)

        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)

        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })
        return result

    # ==================== TREND INDICATORS ====================

    def sma(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Simple Moving Average (SMA)

        Args:
            period: Lookback period
            column: Price column to use

        Returns:
            SMA values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].rolling(window=period).mean()
        else:
            return self.data[column].rolling(window=period).mean()

    def ema(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Exponential Moving Average (EMA)

        Args:
            period: Lookback period
            column: Price column to use

        Returns:
            EMA values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].ewm(span=period, adjust=False).mean()
        else:
            return self.data[column].ewm(span=period, adjust=False).mean()

    def adx(self, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index (ADX)

        Measures strength of trend (regardless of direction).
        Values >25 indicate strong trend, <20 indicate weak trend.

        Args:
            period: Lookback period (default 14)

        Returns:
            DataFrame with columns: ['adx', '+di', '-di']
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_adx(x, period)
            )
        else:
            return self._calculate_adx(self.data, period)

    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """Calculate ADX for a single stock."""
        # Calculate +DM and -DM
        high_diff = data['high'].diff()
        low_diff = -data['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate True Range
        atr = self._calculate_atr(data, period)

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        result = pd.DataFrame({
            'adx': adx,
            '+di': plus_di,
            '-di': minus_di
        })
        return result

    # ==================== VOLUME INDICATORS ====================

    def obv(self) -> pd.Series:
        """
        On-Balance Volume (OBV)

        Uses volume flow to predict changes in stock price.

        Returns:
            OBV values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_obv(x)
            )
        else:
            return self._calculate_obv(self.data)

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate OBV for a single stock."""
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        return obv

    def mfi(self, period: int = 14) -> pd.Series:
        """
        Money Flow Index (MFI)

        Volume-weighted RSI. Measures buying/selling pressure.
        Range: 0-100 (>80 overbought, <20 oversold)

        Args:
            period: Lookback period (default 14)

        Returns:
            MFI values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_mfi(x, period)
            )
        else:
            return self._calculate_mfi(self.data, period)

    def _calculate_mfi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate MFI for a single stock."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']

        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)

        # Money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfr = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mfr))

        return mfi

    def vwap(self) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)

        Average price weighted by volume (typically calculated intraday).
        For daily data, this is a cumulative measure.

        Returns:
            VWAP values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0).apply(
                lambda x: self._calculate_vwap(x)
            )
        else:
            return self._calculate_vwap(self.data)

    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP for a single stock."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap

    # ==================== STATISTICAL MEASURES ====================

    def geometric_return(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Geometric Mean Return

        Compounded average rate of return over period.
        More accurate for volatile assets than arithmetic mean.

        Args:
            period: Lookback period
            column: Price column to use

        Returns:
            Geometric return
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].apply(
                lambda x: self._calculate_geometric_return(x, period)
            )
        else:
            return self._calculate_geometric_return(self.data[column], period)

    def _calculate_geometric_return(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate geometric return for a single price series."""
        returns = prices.pct_change()
        geometric_return = (1 + returns).rolling(window=period).apply(
            lambda x: np.prod(x) ** (1 / len(x)) - 1, raw=True
        )
        return geometric_return

    def arithmetic_return(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Arithmetic Mean Return

        Simple average of returns over period.

        Args:
            period: Lookback period
            column: Price column to use

        Returns:
            Arithmetic return
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].apply(
                lambda x: x.pct_change().rolling(window=period).mean()
            )
        else:
            return self.data[column].pct_change().rolling(window=period).mean()

    def zscore(self, period: int = 20, column: str = 'close') -> pd.Series:
        """
        Z-Score (Rolling)

        Number of standard deviations from the mean.
        Used for mean reversion strategies.

        Args:
            period: Lookback period
            column: Price column to use

        Returns:
            Z-score values
        """
        if self.is_multiindex:
            return self.data.groupby(level=0)[column].apply(
                lambda x: self._calculate_zscore(x, period)
            )
        else:
            return self._calculate_zscore(self.data[column], period)

    def _calculate_zscore(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate z-score for a single price series."""
        mean = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        zscore = (prices - mean) / std
        return zscore

    def volatility(self, period: int = 20, annualize: bool = True, column: str = 'close') -> pd.Series:
        """
        Historical Volatility (Rolling Standard Deviation)

        Args:
            period: Lookback period
            annualize: If True, annualize volatility (multiply by sqrt(252))
            column: Price column to use

        Returns:
            Volatility values
        """
        if self.is_multiindex:
            vol = self.data.groupby(level=0)[column].apply(
                lambda x: x.pct_change().rolling(window=period).std()
            )
        else:
            vol = self.data[column].pct_change().rolling(window=period).std()

        if annualize:
            vol = vol * np.sqrt(252)  # Assuming 252 trading days per year

        return vol

    # ==================== UTILITY METHODS ====================

    def calculate_all_indicators(
        self,
        rsi_period: int = 14,
        macd_params: Tuple[int, int, int] = (12, 26, 9),
        bb_period: int = 20,
        atr_period: int = 14
    ) -> pd.DataFrame:
        """
        Calculate all major technical indicators at once.

        Useful for feature engineering and machine learning.

        Args:
            rsi_period: RSI lookback period
            macd_params: (fast, slow, signal) periods for MACD
            bb_period: Bollinger Bands period
            atr_period: ATR period

        Returns:
            DataFrame with all indicators as columns
        """
        results = {}

        # Momentum
        results['rsi'] = self.rsi(rsi_period)
        macd_df = self.macd(*macd_params)
        results['macd'] = macd_df['macd']
        results['macd_signal'] = macd_df['signal']
        results['macd_histogram'] = macd_df['histogram']

        # Volatility
        bb_df = self.bollinger_bands(bb_period)
        results['bb_upper'] = bb_df['upper']
        results['bb_middle'] = bb_df['middle']
        results['bb_lower'] = bb_df['lower']
        results['bb_bandwidth'] = bb_df['bandwidth']
        results['bb_pct_b'] = bb_df['pct_b']
        results['atr'] = self.atr(atr_period)

        # Trend
        results['sma_20'] = self.sma(20)
        results['sma_50'] = self.sma(50)
        results['ema_12'] = self.ema(12)
        results['ema_26'] = self.ema(26)

        # Volume
        results['obv'] = self.obv()
        results['mfi'] = self.mfi(14)

        # Statistical
        results['volatility_20d'] = self.volatility(20)
        results['geometric_return_20d'] = self.geometric_return(20)
        results['arithmetic_return_20d'] = self.arithmetic_return(20)

        return pd.DataFrame(results)


def add_all_indicators(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Convenience function to add all technical indicators to a DataFrame.

    Args:
        data: OHLCV DataFrame
        **kwargs: Parameters to pass to calculate_all_indicators()

    Returns:
        Original DataFrame with indicator columns added
    """
    indicators = TechnicalIndicators(data)
    indicator_df = indicators.calculate_all_indicators(**kwargs)

    return pd.concat([data, indicator_df], axis=1)
