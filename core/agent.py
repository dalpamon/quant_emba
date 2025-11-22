"""
Agentic Strategy Explorer

Autonomous strategy discovery and optimization for quantitative trading.
The "agent" autonomously:
- Explores parameter space
- Generates and tests strategy variations
- Learns from results
- Adapts search strategy
- Discovers optimal parameter combinations

Designed for Korean stock market trading.

Author: Agentic Backtesting Framework
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from datetime import datetime

from core.feature_engineering import FeatureEngineeringPipeline, FeatureConfig, FeatureLever
from core.optimizer import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    GeneticAlgorithmOptimizer,
    WalkForwardOptimizer,
    OptimizationResult
)
from core.backtest import Backtester
from core.portfolio import Portfolio
from core.analytics import PerformanceAnalytics


@dataclass
class StrategyCandidate:
    """
    Represents a strategy discovered by the agent.

    Attributes:
        name: Strategy identifier
        config: Feature engineering configuration
        performance: Backtest performance metrics
        parameters: Strategy parameters (feature weights, etc.)
        risk_metrics: Risk-adjusted performance metrics
        timestamp: When strategy was discovered
    """
    name: str
    config: FeatureConfig
    performance: Dict[str, float]
    parameters: Dict[str, Any]
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def sharpe_ratio(self) -> float:
        """Get Sharpe ratio (primary optimization target)."""
        return self.risk_metrics.get('sharpe_ratio', -np.inf)

    def sortino_ratio(self) -> float:
        """Get Sortino ratio."""
        return self.risk_metrics.get('sortino_ratio', -np.inf)

    def calmar_ratio(self) -> float:
        """Get Calmar ratio."""
        return self.risk_metrics.get('calmar_ratio', -np.inf)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'config': self.config.to_dict(),
            'performance': self.performance,
            'parameters': self.parameters,
            'risk_metrics': self.risk_metrics,
            'timestamp': self.timestamp
        }


class AgenticStrategyExplorer:
    """
    Autonomous strategy explorer using agentic optimization.

    The agent:
    1. Explores parameter space systematically
    2. Learns which parameter combinations work well
    3. Focuses search on promising regions
    4. Discovers multiple high-performing strategies
    5. Adapts to market characteristics (e.g., Korean market mean reversion)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        transaction_cost_bps: float = 10,
        rebalance_freq: str = 'M',
        objective: str = 'sharpe',
        korean_market: bool = True
    ):
        """
        Initialize agentic explorer.

        Args:
            data: OHLCV price data (MultiIndex with ticker, date)
            initial_capital: Starting capital
            transaction_cost_bps: Transaction cost in basis points
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
            objective: Optimization objective ('sharpe', 'sortino', 'calmar', 'returns')
            korean_market: If True, apply Korean market adaptations
        """
        self.data = data
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.rebalance_freq = rebalance_freq
        self.objective = objective
        self.korean_market = korean_market

        # Initialize feature engineering pipeline
        self.pipeline = FeatureEngineeringPipeline(data)

        # Storage for discovered strategies
        self.discovered_strategies: List[StrategyCandidate] = []
        self.optimization_history: List[OptimizationResult] = []

        # Agent memory (for learning)
        self.parameter_performance_map: Dict[str, List[float]] = {}

        print(f"🤖 Agentic Strategy Explorer initialized")
        print(f"   Data shape: {data.shape}")
        print(f"   Objective: {objective}")
        print(f"   Korean market mode: {korean_market}")

    def explore(
        self,
        method: str = 'genetic',
        n_strategies: int = 10,
        exploration_budget: int = 200,
        verbose: bool = True
    ) -> List[StrategyCandidate]:
        """
        Autonomously explore parameter space and discover strategies.

        Args:
            method: Optimization method ('grid', 'random', 'genetic')
            n_strategies: Number of diverse strategies to discover
            exploration_budget: Total evaluations budget
            verbose: Print progress

        Returns:
            List of discovered StrategyCandidate objects
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🤖 AGENTIC EXPLORATION STARTED")
            print(f"{'='*70}")
            print(f"Method: {method}")
            print(f"Target Strategies: {n_strategies}")
            print(f"Evaluation Budget: {exploration_budget}")
            print(f"{'='*70}\n")

        start_time = time.time()

        # Phase 1: Broad exploration
        if verbose:
            print("📊 Phase 1: Broad Exploration")

        broad_result = self._broad_exploration(
            method=method,
            n_iter=exploration_budget // 2,
            verbose=verbose
        )

        self.optimization_history.append(broad_result)

        # Phase 2: Local refinement around best candidates
        if verbose:
            print("\n🎯 Phase 2: Local Refinement")

        refined_results = self._local_refinement(
            top_n=n_strategies,
            n_iter=exploration_budget // (2 * n_strategies),
            verbose=verbose
        )

        self.optimization_history.extend(refined_results)

        # Phase 3: Diversity selection
        if verbose:
            print("\n🌈 Phase 3: Diversity Selection")

        diverse_strategies = self._select_diverse_strategies(n_strategies)

        exploration_time = time.time() - start_time

        if verbose:
            print(f"\n{'='*70}")
            print(f"✅ EXPLORATION COMPLETE")
            print(f"{'='*70}")
            print(f"Time: {exploration_time:.2f} seconds")
            print(f"Strategies discovered: {len(diverse_strategies)}")
            print(f"Total evaluations: {sum(r.iterations for r in self.optimization_history)}")
            print(f"{'='*70}\n")

            # Show top strategies
            print("🏆 Top Strategies:")
            for i, strategy in enumerate(diverse_strategies[:5], 1):
                sharpe = strategy.sharpe_ratio()
                returns = strategy.performance.get('total_return', 0)
                print(f"  {i}. {strategy.name}: Sharpe={sharpe:.3f}, Return={returns:.2%}")

        return diverse_strategies

    def _broad_exploration(
        self,
        method: str,
        n_iter: int,
        verbose: bool
    ) -> OptimizationResult:
        """Phase 1: Broad exploration of parameter space."""

        # Define objective function
        def objective_func(params: Dict) -> float:
            return self._evaluate_strategy(params, verbose=False)

        # Get levers to optimize
        levers = self.pipeline.levers

        # Run optimization
        if method == 'grid':
            optimizer = GridSearchOptimizer(objective_func, levers, maximize=True)
            result = optimizer.optimize(verbose=verbose)
        elif method == 'random':
            optimizer = RandomSearchOptimizer(objective_func, levers, maximize=True)
            result = optimizer.optimize(n_iter=n_iter, verbose=verbose)
        elif method == 'genetic':
            optimizer = GeneticAlgorithmOptimizer(objective_func, levers, maximize=True)
            result = optimizer.optimize(
                population_size=min(50, n_iter // 4),
                n_generations=20,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Store best strategy
        best_strategy = self._create_strategy_from_params(
            params=result.best_config,
            name="Broad_Exploration_Best",
            score=result.best_score
        )
        self.discovered_strategies.append(best_strategy)

        return result

    def _local_refinement(
        self,
        top_n: int,
        n_iter: int,
        verbose: bool
    ) -> List[OptimizationResult]:
        """Phase 2: Refine around promising parameter regions."""

        # Get top performers from broad exploration
        if not self.optimization_history:
            return []

        broad_result = self.optimization_history[-1]
        results_df = broad_result.to_dataframe()

        if len(results_df) < top_n:
            top_configs = results_df.head(len(results_df))
        else:
            top_configs = results_df.head(top_n)

        refinement_results = []

        for i, row in enumerate(top_configs.iterrows(), 1):
            if verbose:
                print(f"  Refining region {i}/{top_n}...")

            config_dict = row[1].to_dict()
            base_params = {k: v for k, v in config_dict.items() if k != broad_result.objective_name}

            # Create narrowed levers around this region
            narrow_levers = self._create_narrow_levers(base_params)

            # Define objective
            def objective_func(params: Dict) -> float:
                return self._evaluate_strategy(params, verbose=False)

            # Refine using random search
            optimizer = RandomSearchOptimizer(objective_func, narrow_levers, maximize=True)
            result = optimizer.optimize(n_iter=n_iter, verbose=False)

            refinement_results.append(result)

            # Store strategy
            strategy = self._create_strategy_from_params(
                params=result.best_config,
                name=f"Refined_Region_{i}",
                score=result.best_score
            )
            self.discovered_strategies.append(strategy)

        return refinement_results

    def _create_narrow_levers(
        self,
        base_params: Dict[str, Any],
        search_width: float = 0.2
    ) -> Dict[str, FeatureLever]:
        """Create narrowed parameter levers around base configuration."""

        narrow_levers = {}

        for param_name, base_value in base_params.items():
            if param_name not in self.pipeline.levers:
                continue

            original_lever = self.pipeline.levers[param_name]

            if original_lever.data_type in ['int', 'float']:
                # Narrow the range around base value
                range_size = original_lever.max_val - original_lever.min_val
                delta = range_size * search_width / 2

                if original_lever.data_type == 'int':
                    new_min = max(original_lever.min_val, int(base_value - delta))
                    new_max = min(original_lever.max_val, int(base_value + delta))
                else:
                    new_min = max(original_lever.min_val, base_value - delta)
                    new_max = min(original_lever.max_val, base_value + delta)

                narrow_levers[param_name] = FeatureLever(
                    name=param_name,
                    default=base_value,
                    min_val=new_min,
                    max_val=new_max,
                    step=original_lever.step,
                    data_type=original_lever.data_type,
                    description=original_lever.description
                )
            else:
                # Keep original for categorical/bool
                narrow_levers[param_name] = original_lever

        return narrow_levers

    def _select_diverse_strategies(self, n_strategies: int) -> List[StrategyCandidate]:
        """Phase 3: Select diverse, high-performing strategies."""

        if len(self.discovered_strategies) <= n_strategies:
            return sorted(
                self.discovered_strategies,
                key=lambda s: s.sharpe_ratio(),
                reverse=True
            )

        # Start with best strategy
        selected = [max(self.discovered_strategies, key=lambda s: s.sharpe_ratio())]

        # Greedily add strategies that are diverse and high-performing
        candidates = [s for s in self.discovered_strategies if s not in selected]

        while len(selected) < n_strategies and candidates:
            # Score each candidate by: performance + diversity
            best_candidate = None
            best_score = -np.inf

            for candidate in candidates:
                performance_score = candidate.sharpe_ratio()
                diversity_score = self._diversity_score(candidate, selected)

                # Combined score (50% performance, 50% diversity)
                combined_score = 0.5 * performance_score + 0.5 * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break

        return selected

    def _diversity_score(
        self,
        candidate: StrategyCandidate,
        selected: List[StrategyCandidate]
    ) -> float:
        """Calculate how diverse candidate is from selected strategies."""

        if not selected:
            return 1.0

        # Calculate parameter distances
        distances = []

        for selected_strategy in selected:
            param_distance = 0
            n_params = 0

            for param_name in candidate.parameters:
                if param_name in selected_strategy.parameters:
                    # Normalize parameter values
                    lever = self.pipeline.levers.get(param_name)
                    if lever and lever.data_type in ['int', 'float']:
                        range_size = lever.max_val - lever.min_val
                        if range_size > 0:
                            val1 = (candidate.parameters[param_name] - lever.min_val) / range_size
                            val2 = (selected_strategy.parameters[param_name] - lever.min_val) / range_size
                            param_distance += abs(val1 - val2)
                            n_params += 1

            if n_params > 0:
                distances.append(param_distance / n_params)

        # Return average distance (higher = more diverse)
        return np.mean(distances) if distances else 0

    def _evaluate_strategy(self, params: Dict[str, Any], verbose: bool = False) -> float:
        """
        Evaluate strategy with given parameters.

        Returns objective function value (e.g., Sharpe ratio).
        """
        try:
            # Set pipeline parameters
            for param_name, param_value in params.items():
                self.pipeline.set_lever(param_name, param_value)

            # Generate features
            features = self.pipeline.generate_features()

            # For simplicity, use equal-weighted composite of all features
            # In practice, could optimize feature weights too
            composite_signal = features.mean(axis=1)

            # Run backtest
            performance = self._backtest_signal(composite_signal)

            # Return objective
            if self.objective == 'sharpe':
                return performance.get('sharpe_ratio', -np.inf)
            elif self.objective == 'sortino':
                return performance.get('sortino_ratio', -np.inf)
            elif self.objective == 'calmar':
                return performance.get('calmar_ratio', -np.inf)
            elif self.objective == 'returns':
                return performance.get('total_return', -np.inf)
            else:
                return performance.get('sharpe_ratio', -np.inf)

        except Exception as e:
            if verbose:
                print(f"Error evaluating strategy: {e}")
            return -np.inf

    def _backtest_signal(self, signal: pd.Series) -> Dict[str, float]:
        """
        Backtest a trading signal.

        Args:
            signal: Trading signal (higher = more bullish)

        Returns:
            Performance metrics dictionary
        """
        # Create long-short portfolio based on signal
        # Top 20% long, bottom 20% short
        if isinstance(signal.index, pd.MultiIndex):
            # Group by date and rank
            ranks = signal.groupby(level=1).rank(pct=True)
        else:
            ranks = signal.rank(pct=True)

        # Create positions: 1 for long, -1 for short, 0 for neutral
        positions = pd.Series(0, index=ranks.index)
        positions[ranks >= 0.8] = 1  # Top 20% long
        positions[ranks <= 0.2] = -1  # Bottom 20% short

        # Reshape for backtester
        if isinstance(self.data.index, pd.MultiIndex):
            position_schedule = positions.unstack(level=0)
        else:
            position_schedule = positions.to_frame()

        # Run backtest
        backtester = Backtester(
            initial_capital=self.initial_capital,
            transaction_cost_bps=self.transaction_cost_bps,
            rebalance_freq=self.rebalance_freq
        )

        results = backtester.run(self.data, position_schedule)

        # Calculate analytics
        analytics = PerformanceAnalytics(results['daily_returns'])

        metrics = {
            'total_return': results['cumulative_returns'].iloc[-1] - 1,
            'sharpe_ratio': analytics.sharpe_ratio(),
            'sortino_ratio': analytics.sortino_ratio(),
            'calmar_ratio': analytics.calmar_ratio(),
            'max_drawdown': analytics.max_drawdown(),
            'volatility': analytics.volatility()
        }

        return metrics

    def _create_strategy_from_params(
        self,
        params: Dict[str, Any],
        name: str,
        score: float
    ) -> StrategyCandidate:
        """Create StrategyCandidate from parameters."""

        # Set parameters
        for param_name, param_value in params.items():
            self.pipeline.set_lever(param_name, param_value)

        # Generate features and evaluate
        features = self.pipeline.generate_features()
        composite_signal = features.mean(axis=1)
        performance = self._backtest_signal(composite_signal)

        # Create config
        config = FeatureConfig(
            name=name,
            levers=params.copy(),
            enabled_features=list(features.columns)
        )

        strategy = StrategyCandidate(
            name=name,
            config=config,
            performance=performance,
            parameters=params,
            risk_metrics=performance
        )

        return strategy

    def walk_forward_validate(
        self,
        strategy: StrategyCandidate,
        in_sample_months: int = 12,
        out_sample_months: int = 3,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Validate strategy using walk-forward analysis.

        Args:
            strategy: Strategy to validate
            in_sample_months: Training period months
            out_sample_months: Testing period months
            verbose: Print progress

        Returns:
            DataFrame with walk-forward results
        """
        # Create objective function for this strategy
        def objective_func(params: Dict, data: pd.DataFrame) -> float:
            # Temporarily set data
            original_data = self.data
            self.data = data

            score = self._evaluate_strategy(params, verbose=False)

            # Restore original data
            self.data = original_data

            return score

        # Run walk-forward optimization
        wf_optimizer = WalkForwardOptimizer(
            data=self.data,
            objective_function=objective_func,
            levers=self.pipeline.levers,
            optimizer_class=RandomSearchOptimizer,
            in_sample_months=in_sample_months,
            out_sample_months=out_sample_months,
            maximize=True
        )

        results = wf_optimizer.optimize(
            optimizer_kwargs={'n_iter': 50},
            verbose=verbose
        )

        return results

    def save_strategies(self, filepath: str):
        """Save discovered strategies to JSON file."""
        data = [strategy.to_dict() for strategy in self.discovered_strategies]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"💾 Saved {len(data)} strategies to {filepath}")

    def export_best_strategy(self, output_dir: str = 'output/strategies'):
        """Export best strategy for production use."""
        if not self.discovered_strategies:
            print("No strategies to export")
            return

        best = max(self.discovered_strategies, key=lambda s: s.sharpe_ratio())

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save configuration
        config_path = f"{output_dir}/best_strategy_config.json"
        best.config.save(config_path)

        # Save full strategy
        strategy_path = f"{output_dir}/best_strategy.json"
        with open(strategy_path, 'w') as f:
            json.dump(best.to_dict(), f, indent=2)

        print(f"📦 Exported best strategy:")
        print(f"   Name: {best.name}")
        print(f"   Sharpe: {best.sharpe_ratio():.3f}")
        print(f"   Config: {config_path}")
        print(f"   Full: {strategy_path}")

    def get_strategy_comparison(self) -> pd.DataFrame:
        """
        Compare all discovered strategies.

        Returns:
            DataFrame with strategy metrics
        """
        if not self.discovered_strategies:
            return pd.DataFrame()

        data = []
        for strategy in self.discovered_strategies:
            row = {
                'Strategy': strategy.name,
                'Sharpe': strategy.sharpe_ratio(),
                'Sortino': strategy.sortino_ratio(),
                'Calmar': strategy.calmar_ratio(),
                'Total Return': strategy.performance.get('total_return', 0),
                'Max Drawdown': strategy.performance.get('max_drawdown', 0),
                'Volatility': strategy.performance.get('volatility', 0)
            }
            data.append(row)

        df = pd.DataFrame(data)
        return df.sort_values('Sharpe', ascending=False)


def quick_explore(
    data: pd.DataFrame,
    n_strategies: int = 5,
    budget: int = 100,
    objective: str = 'sharpe',
    korean_market: bool = True
) -> AgenticStrategyExplorer:
    """
    Quick agentic exploration with default settings.

    Args:
        data: OHLCV data
        n_strategies: Number of strategies to discover
        budget: Evaluation budget
        objective: Optimization objective
        korean_market: Korean market mode

    Returns:
        AgenticStrategyExplorer with discovered strategies
    """
    explorer = AgenticStrategyExplorer(
        data=data,
        objective=objective,
        korean_market=korean_market
    )

    explorer.explore(
        method='genetic',
        n_strategies=n_strategies,
        exploration_budget=budget,
        verbose=True
    )

    return explorer
