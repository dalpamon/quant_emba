"""
Feature Engineering Pipeline for Quantitative Trading

Provides flexible, configurable feature generation with "levers" for parameter tuning.
Designed for agentic optimization and Korean stock market trading.

Author: Agentic Backtesting Framework
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

from core.indicators import TechnicalIndicators


@dataclass
class FeatureLever:
    """
    Represents a configurable parameter (lever) for feature engineering.

    Attributes:
        name: Lever identifier
        default: Default value
        min_val: Minimum value
        max_val: Maximum value
        step: Step size for optimization
        description: Human-readable description
        data_type: 'int', 'float', 'bool', or 'categorical'
        options: For categorical levers, list of valid options
    """
    name: str
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    description: str = ""
    data_type: str = "float"  # 'int', 'float', 'bool', 'categorical'
    options: Optional[List[Any]] = None

    def validate(self, value: Any) -> bool:
        """Validate if value is within lever constraints."""
        if self.data_type == 'categorical':
            return value in self.options
        elif self.data_type == 'bool':
            return isinstance(value, bool)
        elif self.data_type in ['int', 'float']:
            if self.min_val is not None and value < self.min_val:
                return False
            if self.max_val is not None and value > self.max_val:
                return False
            return True
        return True

    def get_optimization_range(self) -> List[Any]:
        """Get list of values to test during optimization."""
        if self.data_type == 'categorical':
            return self.options
        elif self.data_type == 'bool':
            return [True, False]
        elif self.data_type == 'int':
            if self.min_val is not None and self.max_val is not None and self.step is not None:
                return list(range(int(self.min_val), int(self.max_val) + 1, int(self.step)))
            return [self.default]
        elif self.data_type == 'float':
            if self.min_val is not None and self.max_val is not None and self.step is not None:
                return list(np.arange(self.min_val, self.max_val + self.step, self.step))
            return [self.default]
        return [self.default]


@dataclass
class FeatureConfig:
    """
    Configuration for a feature engineering pipeline.

    Attributes:
        name: Configuration name
        levers: Dictionary of lever_name -> current_value
        enabled_features: List of feature names to compute
        metadata: Additional metadata
    """
    name: str
    levers: Dict[str, Any] = field(default_factory=dict)
    enabled_features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'levers': self.levers,
            'enabled_features': self.enabled_features,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureConfig':
        """Deserialize from dictionary."""
        return cls(
            name=data['name'],
            levers=data.get('levers', {}),
            enabled_features=data.get('enabled_features', []),
            metadata=data.get('metadata', {})
        )

    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'FeatureConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class FeatureEngineeringPipeline:
    """
    Flexible feature engineering pipeline with configurable levers.

    Supports:
    - Technical indicators with tunable parameters
    - Custom feature functions
    - Feature selection and normalization
    - Configuration save/load for reproducibility
    - Integration with optimization engines
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize pipeline with OHLCV data.

        Args:
            data: DataFrame with OHLCV columns (can be MultiIndex)
        """
        self.data = data.copy()
        self.indicators = TechnicalIndicators(data)
        self.levers: Dict[str, FeatureLever] = {}
        self.custom_features: Dict[str, Callable] = {}
        self.config = FeatureConfig(name="default")

        # Initialize default levers
        self._initialize_default_levers()

    def _initialize_default_levers(self):
        """Set up default parameter levers for common indicators."""

        # RSI parameters
        self.add_lever(FeatureLever(
            name="rsi_period",
            default=14,
            min_val=5,
            max_val=30,
            step=1,
            data_type="int",
            description="RSI lookback period"
        ))

        # MACD parameters
        self.add_lever(FeatureLever(
            name="macd_fast",
            default=12,
            min_val=8,
            max_val=20,
            step=2,
            data_type="int",
            description="MACD fast EMA period"
        ))
        self.add_lever(FeatureLever(
            name="macd_slow",
            default=26,
            min_val=20,
            max_val=40,
            step=2,
            data_type="int",
            description="MACD slow EMA period"
        ))
        self.add_lever(FeatureLever(
            name="macd_signal",
            default=9,
            min_val=5,
            max_val=15,
            step=1,
            data_type="int",
            description="MACD signal period"
        ))

        # Bollinger Bands parameters
        self.add_lever(FeatureLever(
            name="bb_period",
            default=20,
            min_val=10,
            max_val=50,
            step=5,
            data_type="int",
            description="Bollinger Bands period"
        ))
        self.add_lever(FeatureLever(
            name="bb_std",
            default=2.0,
            min_val=1.0,
            max_val=3.0,
            step=0.5,
            data_type="float",
            description="Bollinger Bands std deviation"
        ))

        # Moving average parameters
        self.add_lever(FeatureLever(
            name="sma_short",
            default=20,
            min_val=5,
            max_val=50,
            step=5,
            data_type="int",
            description="Short-term SMA period"
        ))
        self.add_lever(FeatureLever(
            name="sma_long",
            default=50,
            min_val=30,
            max_val=200,
            step=10,
            data_type="int",
            description="Long-term SMA period"
        ))

        # ATR parameters
        self.add_lever(FeatureLever(
            name="atr_period",
            default=14,
            min_val=7,
            max_val=21,
            step=1,
            data_type="int",
            description="ATR period"
        ))

        # Momentum parameters
        self.add_lever(FeatureLever(
            name="momentum_period",
            default=20,
            min_val=5,
            max_val=60,
            step=5,
            data_type="int",
            description="Momentum lookback period"
        ))

        # Volatility parameters
        self.add_lever(FeatureLever(
            name="volatility_period",
            default=20,
            min_val=10,
            max_val=60,
            step=5,
            data_type="int",
            description="Volatility calculation period"
        ))

        # Stochastic parameters
        self.add_lever(FeatureLever(
            name="stochastic_k",
            default=14,
            min_val=5,
            max_val=21,
            step=1,
            data_type="int",
            description="Stochastic %K period"
        ))
        self.add_lever(FeatureLever(
            name="stochastic_d",
            default=3,
            min_val=2,
            max_val=5,
            step=1,
            data_type="int",
            description="Stochastic %D smoothing"
        ))

        # Volume indicators
        self.add_lever(FeatureLever(
            name="mfi_period",
            default=14,
            min_val=7,
            max_val=21,
            step=1,
            data_type="int",
            description="Money Flow Index period"
        ))

        # Feature normalization
        self.add_lever(FeatureLever(
            name="normalize_features",
            default=True,
            data_type="bool",
            description="Apply z-score normalization to features"
        ))

        # Feature selection
        self.add_lever(FeatureLever(
            name="feature_selection_method",
            default="all",
            data_type="categorical",
            options=["all", "momentum_only", "volatility_only", "volume_only", "custom"],
            description="Which features to include"
        ))

    def add_lever(self, lever: FeatureLever):
        """Add a configurable lever to the pipeline."""
        self.levers[lever.name] = lever
        if lever.name not in self.config.levers:
            self.config.levers[lever.name] = lever.default

    def set_lever(self, name: str, value: Any):
        """Set lever value with validation."""
        if name not in self.levers:
            raise ValueError(f"Lever '{name}' not found")

        lever = self.levers[name]
        if not lever.validate(value):
            raise ValueError(
                f"Invalid value {value} for lever '{name}'. "
                f"Constraints: min={lever.min_val}, max={lever.max_val}"
            )

        self.config.levers[name] = value

    def get_lever(self, name: str) -> Any:
        """Get current lever value."""
        return self.config.levers.get(name, self.levers[name].default)

    def add_custom_feature(self, name: str, func: Callable[[pd.DataFrame], pd.Series]):
        """
        Add custom feature function.

        Args:
            name: Feature name
            func: Function that takes OHLCV DataFrame and returns Series
        """
        self.custom_features[name] = func

    def generate_features(self) -> pd.DataFrame:
        """
        Generate all features based on current lever settings.

        Returns:
            DataFrame with all computed features
        """
        features = {}

        # Get current lever values
        rsi_period = self.get_lever("rsi_period")
        macd_fast = self.get_lever("macd_fast")
        macd_slow = self.get_lever("macd_slow")
        macd_signal = self.get_lever("macd_signal")
        bb_period = self.get_lever("bb_period")
        bb_std = self.get_lever("bb_std")
        sma_short = self.get_lever("sma_short")
        sma_long = self.get_lever("sma_long")
        atr_period = self.get_lever("atr_period")
        momentum_period = self.get_lever("momentum_period")
        volatility_period = self.get_lever("volatility_period")
        stochastic_k = self.get_lever("stochastic_k")
        stochastic_d = self.get_lever("stochastic_d")
        mfi_period = self.get_lever("mfi_period")
        selection_method = self.get_lever("feature_selection_method")

        # Generate features based on selection method
        if selection_method in ["all", "momentum_only"]:
            # RSI
            features['rsi'] = self.indicators.rsi(rsi_period)

            # MACD
            macd_df = self.indicators.macd(macd_fast, macd_slow, macd_signal)
            features['macd'] = macd_df['macd']
            features['macd_signal'] = macd_df['signal']
            features['macd_histogram'] = macd_df['histogram']

            # Stochastic
            stoch_df = self.indicators.stochastic(stochastic_k, stochastic_d)
            features['stochastic_k'] = stoch_df['%K']
            features['stochastic_d'] = stoch_df['%D']

            # Williams %R
            features['williams_r'] = self.indicators.williams_r(rsi_period)

            # Momentum (price returns)
            features['momentum'] = self.indicators.arithmetic_return(momentum_period)
            features['geometric_momentum'] = self.indicators.geometric_return(momentum_period)

        if selection_method in ["all", "volatility_only"]:
            # Bollinger Bands
            bb_df = self.indicators.bollinger_bands(bb_period, bb_std)
            features['bb_upper'] = bb_df['upper']
            features['bb_middle'] = bb_df['middle']
            features['bb_lower'] = bb_df['lower']
            features['bb_bandwidth'] = bb_df['bandwidth']
            features['bb_pct_b'] = bb_df['pct_b']

            # ATR
            features['atr'] = self.indicators.atr(atr_period)

            # Volatility
            features['volatility'] = self.indicators.volatility(volatility_period)

            # Z-score (mean reversion indicator)
            features['zscore'] = self.indicators.zscore(bb_period)

        if selection_method in ["all", "volume_only"]:
            # OBV
            features['obv'] = self.indicators.obv()

            # MFI
            features['mfi'] = self.indicators.mfi(mfi_period)

            # VWAP
            features['vwap'] = self.indicators.vwap()

        if selection_method == "all":
            # Trend indicators
            features['sma_short'] = self.indicators.sma(sma_short)
            features['sma_long'] = self.indicators.sma(sma_long)
            features['ema_12'] = self.indicators.ema(12)
            features['ema_26'] = self.indicators.ema(26)

            # Price position relative to MAs
            if isinstance(self.data.index, pd.MultiIndex):
                close_prices = self.data['close']
            else:
                close_prices = self.data['close']

            features['price_vs_sma_short'] = (close_prices - features['sma_short']) / features['sma_short']
            features['price_vs_sma_long'] = (close_prices - features['sma_long']) / features['sma_long']

            # ADX (trend strength)
            adx_df = self.indicators.adx(rsi_period)
            features['adx'] = adx_df['adx']
            features['plus_di'] = adx_df['+di']
            features['minus_di'] = adx_df['-di']

        # Add custom features
        for feat_name, feat_func in self.custom_features.items():
            try:
                features[feat_name] = feat_func(self.data)
            except Exception as e:
                print(f"Warning: Failed to compute custom feature '{feat_name}': {e}")

        # Combine into DataFrame
        feature_df = pd.DataFrame(features)

        # Normalize if enabled
        if self.get_lever("normalize_features"):
            feature_df = self._normalize_features(feature_df)

        return feature_df

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score normalization to features.

        Args:
            features: DataFrame with features

        Returns:
            Normalized features
        """
        if isinstance(features.index, pd.MultiIndex):
            # Group by date for cross-sectional normalization
            normalized = features.groupby(level=1).apply(
                lambda x: (x - x.mean()) / x.std()
            )
        else:
            # Time-series normalization
            normalized = (features - features.mean()) / features.std()

        return normalized.fillna(0)

    def create_composite_signal(
        self,
        feature_weights: Dict[str, float],
        features: Optional[pd.DataFrame] = None
    ) -> pd.Series:
        """
        Create composite signal from weighted features.

        Args:
            feature_weights: Dictionary of feature_name -> weight
            features: Pre-computed features (if None, will generate)

        Returns:
            Composite signal (weighted sum of features)
        """
        if features is None:
            features = self.generate_features()

        # Validate features exist
        missing_features = set(feature_weights.keys()) - set(features.columns)
        if missing_features:
            raise ValueError(f"Features not found: {missing_features}")

        # Calculate weighted sum
        signal = pd.Series(0, index=features.index)
        for feat_name, weight in feature_weights.items():
            signal += features[feat_name] * weight

        return signal

    def save_config(self, filepath: str):
        """Save current configuration to file."""
        self.config.save(filepath)
        print(f"Configuration saved to {filepath}")

    def load_config(self, filepath: str):
        """Load configuration from file."""
        self.config = FeatureConfig.load(filepath)

        # Apply loaded lever values
        for lever_name, value in self.config.levers.items():
            if lever_name in self.levers:
                self.set_lever(lever_name, value)

        print(f"Configuration loaded from {filepath}")

    def get_lever_summary(self) -> pd.DataFrame:
        """
        Get summary of all levers and their current values.

        Returns:
            DataFrame with lever information
        """
        data = []
        for name, lever in self.levers.items():
            data.append({
                'Lever': name,
                'Current Value': self.get_lever(name),
                'Default': lever.default,
                'Min': lever.min_val,
                'Max': lever.max_val,
                'Type': lever.data_type,
                'Description': lever.description
            })

        return pd.DataFrame(data)

    def clone_with_config(self, config: FeatureConfig) -> 'FeatureEngineeringPipeline':
        """
        Create a clone of this pipeline with different configuration.

        Useful for parallel optimization runs.

        Args:
            config: New configuration to use

        Returns:
            New pipeline instance
        """
        new_pipeline = FeatureEngineeringPipeline(self.data)
        new_pipeline.config = config
        new_pipeline.levers = self.levers.copy()
        new_pipeline.custom_features = self.custom_features.copy()

        return new_pipeline


class FeatureSelector:
    """
    Statistical feature selection for reducing dimensionality.

    Methods:
    - Correlation-based selection
    - Information coefficient ranking
    - Forward/backward selection based on backtest performance
    """

    @staticmethod
    def remove_correlated_features(
        features: pd.DataFrame,
        threshold: float = 0.8
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.

        Args:
            features: Feature DataFrame
            threshold: Correlation threshold (default 0.8)

        Returns:
            Tuple of (reduced features, list of removed features)
        """
        corr_matrix = features.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        reduced_features = features.drop(columns=to_drop)

        return reduced_features, to_drop

    @staticmethod
    def rank_features_by_ic(
        features: pd.DataFrame,
        forward_returns: pd.Series,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank features by Information Coefficient (IC).

        IC measures correlation between feature and forward returns.

        Args:
            features: Feature DataFrame
            forward_returns: Forward-looking returns
            top_n: Return only top N features (None = all)

        Returns:
            DataFrame with features ranked by absolute IC
        """
        ic_scores = {}

        for col in features.columns:
            # Calculate correlation (IC)
            ic = features[col].corr(forward_returns)
            ic_scores[col] = ic

        ic_df = pd.DataFrame({
            'Feature': ic_scores.keys(),
            'IC': ic_scores.values()
        })

        ic_df['Abs_IC'] = ic_df['IC'].abs()
        ic_df = ic_df.sort_values('Abs_IC', ascending=False)

        if top_n is not None:
            ic_df = ic_df.head(top_n)

        return ic_df


# Convenience functions
def create_default_pipeline(data: pd.DataFrame) -> FeatureEngineeringPipeline:
    """Create pipeline with default settings."""
    return FeatureEngineeringPipeline(data)


def quick_features(
    data: pd.DataFrame,
    feature_set: str = "all"
) -> pd.DataFrame:
    """
    Quickly generate features with default parameters.

    Args:
        data: OHLCV DataFrame
        feature_set: "all", "momentum_only", "volatility_only", or "volume_only"

    Returns:
        Features DataFrame
    """
    pipeline = FeatureEngineeringPipeline(data)
    pipeline.set_lever("feature_selection_method", feature_set)
    return pipeline.generate_features()
