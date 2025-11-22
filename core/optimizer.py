"""
Optimization Engine for Agentic Backtesting

Provides multiple optimization algorithms for finding optimal strategy parameters:
- Grid Search: Exhaustive search over parameter space
- Random Search: Random sampling for high-dimensional spaces
- Genetic Algorithm: Evolutionary optimization
- Bayesian Optimization: Sample-efficient optimization using Gaussian Processes

Author: Agentic Backtesting Framework
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
from dataclasses import dataclass, field
from itertools import product
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

from core.feature_engineering import FeatureConfig, FeatureLever


@dataclass
class OptimizationResult:
    """
    Result from an optimization run.

    Attributes:
        best_config: Best parameter configuration found
        best_score: Best objective function score
        all_results: List of all tested configurations and scores
        optimization_time: Total optimization time in seconds
        iterations: Number of iterations performed
        method: Optimization method used
        objective_name: Name of objective function
    """
    best_config: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict, float]] = field(default_factory=list)
    optimization_time: float = 0.0
    iterations: int = 0
    method: str = "unknown"
    objective_name: str = "objective"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to DataFrame for analysis."""
        if not self.all_results:
            return pd.DataFrame()

        data = []
        for config, score in self.all_results:
            row = config.copy()
            row[self.objective_name] = score
            data.append(row)

        df = pd.DataFrame(data)
        return df.sort_values(self.objective_name, ascending=False)

    def summary(self) -> str:
        """Generate human-readable summary."""
        summary = f"""
Optimization Summary
{'='*60}
Method: {self.method}
Objective: {self.objective_name}
Iterations: {self.iterations}
Time: {self.optimization_time:.2f} seconds
{'='*60}

Best Score: {self.best_score:.4f}

Best Parameters:
{'-'*60}
"""
        for param, value in self.best_config.items():
            summary += f"  {param}: {value}\n"

        summary += f"{'='*60}\n"

        return summary


class ParameterOptimizer:
    """
    Base class for parameter optimization.

    Provides common functionality for all optimization algorithms.
    """

    def __init__(
        self,
        objective_function: Callable[[Dict], float],
        levers: Dict[str, FeatureLever],
        maximize: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize optimizer.

        Args:
            objective_function: Function that takes parameter dict and returns score
            levers: Dictionary of parameter levers to optimize
            maximize: If True, maximize objective; if False, minimize
            random_state: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.levers = levers
        self.maximize = maximize
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        self.history: List[Tuple[Dict, float]] = []

    def _evaluate(self, params: Dict[str, Any]) -> float:
        """Evaluate objective function with given parameters."""
        try:
            score = self.objective_function(params)
            self.history.append((params.copy(), score))
            return score
        except Exception as e:
            print(f"Error evaluating params {params}: {e}")
            return -np.inf if self.maximize else np.inf

    def _is_better(self, score1: float, score2: float) -> bool:
        """Check if score1 is better than score2."""
        if self.maximize:
            return score1 > score2
        else:
            return score1 < score2


class GridSearchOptimizer(ParameterOptimizer):
    """
    Grid Search Optimization.

    Exhaustively searches all combinations of parameter values.
    Best for small parameter spaces.
    """

    def optimize(
        self,
        n_jobs: int = 1,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run grid search optimization.

        Args:
            n_jobs: Number of parallel jobs (-1 = all cores)
            verbose: Print progress

        Returns:
            OptimizationResult object
        """
        import time
        start_time = time.time()

        # Generate parameter grid
        param_grid = self._generate_grid()
        total_combinations = len(param_grid)

        if verbose:
            print(f"Grid Search: Testing {total_combinations} combinations...")

        best_score = -np.inf if self.maximize else np.inf
        best_params = None

        # Evaluate all combinations
        if n_jobs == 1:
            # Sequential execution
            for i, params in enumerate(param_grid):
                score = self._evaluate(params)

                if self._is_better(score, best_score):
                    best_score = score
                    best_params = params

                if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                    print(f"  Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%)")
        else:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
                future_to_params = {
                    executor.submit(self.objective_function, params): params
                    for params in param_grid
                }

                for i, future in enumerate(as_completed(future_to_params)):
                    params = future_to_params[future]
                    try:
                        score = future.result()
                        self.history.append((params, score))

                        if self._is_better(score, best_score):
                            best_score = score
                            best_params = params
                    except Exception as e:
                        print(f"Error with params {params}: {e}")

                    if verbose and (i + 1) % max(1, total_combinations // 10) == 0:
                        print(f"  Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%)")

        optimization_time = time.time() - start_time

        result = OptimizationResult(
            best_config=best_params,
            best_score=best_score,
            all_results=self.history,
            optimization_time=optimization_time,
            iterations=total_combinations,
            method="Grid Search"
        )

        if verbose:
            print(f"\nOptimization complete in {optimization_time:.2f} seconds")
            print(f"Best score: {best_score:.4f}")

        return result

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        param_names = list(self.levers.keys())
        param_ranges = [self.levers[name].get_optimization_range() for name in param_names]

        grid = []
        for combination in product(*param_ranges):
            params = dict(zip(param_names, combination))
            grid.append(params)

        return grid


class RandomSearchOptimizer(ParameterOptimizer):
    """
    Random Search Optimization.

    Randomly samples parameter combinations.
    More efficient than grid search for high-dimensional spaces.
    """

    def optimize(
        self,
        n_iter: int = 100,
        n_jobs: int = 1,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run random search optimization.

        Args:
            n_iter: Number of random samples to test
            n_jobs: Number of parallel jobs
            verbose: Print progress

        Returns:
            OptimizationResult object
        """
        import time
        start_time = time.time()

        if verbose:
            print(f"Random Search: Testing {n_iter} random combinations...")

        best_score = -np.inf if self.maximize else np.inf
        best_params = None

        for i in range(n_iter):
            # Sample random parameters
            params = self._sample_random_params()
            score = self._evaluate(params)

            if self._is_better(score, best_score):
                best_score = score
                best_params = params

            if verbose and (i + 1) % max(1, n_iter // 10) == 0:
                print(f"  Progress: {i+1}/{n_iter} ({100*(i+1)/n_iter:.1f}%) - Best: {best_score:.4f}")

        optimization_time = time.time() - start_time

        result = OptimizationResult(
            best_config=best_params,
            best_score=best_score,
            all_results=self.history,
            optimization_time=optimization_time,
            iterations=n_iter,
            method="Random Search"
        )

        if verbose:
            print(f"\nOptimization complete in {optimization_time:.2f} seconds")

        return result

    def _sample_random_params(self) -> Dict[str, Any]:
        """Sample random parameter values."""
        params = {}

        for name, lever in self.levers.items():
            if lever.data_type == 'categorical':
                params[name] = random.choice(lever.options)
            elif lever.data_type == 'bool':
                params[name] = random.choice([True, False])
            elif lever.data_type == 'int':
                params[name] = random.randint(int(lever.min_val), int(lever.max_val))
            elif lever.data_type == 'float':
                params[name] = random.uniform(lever.min_val, lever.max_val)

        return params


class GeneticAlgorithmOptimizer(ParameterOptimizer):
    """
    Genetic Algorithm Optimization.

    Uses evolutionary principles (selection, crossover, mutation) to find optimal parameters.
    Good for complex, non-convex optimization landscapes.
    """

    def optimize(
        self,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism: float = 0.1,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run genetic algorithm optimization.

        Args:
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
            elitism: Fraction of top individuals to preserve (0-1)
            verbose: Print progress

        Returns:
            OptimizationResult object
        """
        import time
        start_time = time.time()

        if verbose:
            print(f"Genetic Algorithm: {population_size} individuals, {n_generations} generations")

        # Initialize population
        population = [self._sample_random_params() for _ in range(population_size)]

        best_score = -np.inf if self.maximize else np.inf
        best_params = None

        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = [(ind, self._evaluate(ind)) for ind in population]

            # Sort by fitness
            if self.maximize:
                fitness_scores.sort(key=lambda x: x[1], reverse=True)
            else:
                fitness_scores.sort(key=lambda x: x[1])

            # Update best
            gen_best_params, gen_best_score = fitness_scores[0]
            if self._is_better(gen_best_score, best_score):
                best_score = gen_best_score
                best_params = gen_best_params

            if verbose:
                avg_fitness = np.mean([score for _, score in fitness_scores])
                print(f"  Gen {generation+1}/{n_generations}: Best={gen_best_score:.4f}, Avg={avg_fitness:.4f}")

            # Elitism: preserve top individuals
            n_elite = max(1, int(population_size * elitism))
            new_population = [ind for ind, _ in fitness_scores[:n_elite]]

            # Generate rest of new population
            while len(new_population) < population_size:
                # Selection
                parent1 = self._tournament_selection(fitness_scores)
                parent2 = self._tournament_selection(fitness_scores)

                # Crossover
                if random.random() < crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if random.random() < mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

        optimization_time = time.time() - start_time
        total_evals = len(self.history)

        result = OptimizationResult(
            best_config=best_params,
            best_score=best_score,
            all_results=self.history,
            optimization_time=optimization_time,
            iterations=total_evals,
            method="Genetic Algorithm",
            metadata={
                'population_size': population_size,
                'n_generations': n_generations,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate
            }
        )

        if verbose:
            print(f"\nOptimization complete in {optimization_time:.2f} seconds")

        return result

    def _tournament_selection(
        self,
        fitness_scores: List[Tuple[Dict, float]],
        tournament_size: int = 3
    ) -> Dict[str, Any]:
        """Select individual using tournament selection."""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))

        if self.maximize:
            winner = max(tournament, key=lambda x: x[1])
        else:
            winner = min(tournament, key=lambda x: x[1])

        return winner[0]

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Perform crossover between two parents."""
        child = {}

        for param_name in parent1.keys():
            # Randomly choose parameter from either parent
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]

        return child

    def _mutate(self, individual: Dict) -> Dict:
        """Mutate an individual."""
        mutated = individual.copy()

        # Choose random parameter to mutate
        param_name = random.choice(list(mutated.keys()))
        lever = self.levers[param_name]

        if lever.data_type == 'categorical':
            mutated[param_name] = random.choice(lever.options)
        elif lever.data_type == 'bool':
            mutated[param_name] = not mutated[param_name]
        elif lever.data_type == 'int':
            # Add/subtract small amount
            delta = random.randint(-3, 3)
            new_val = mutated[param_name] + delta
            new_val = max(lever.min_val, min(lever.max_val, new_val))
            mutated[param_name] = int(new_val)
        elif lever.data_type == 'float':
            # Add/subtract small percentage
            delta = random.uniform(-0.2, 0.2) * (lever.max_val - lever.min_val)
            new_val = mutated[param_name] + delta
            new_val = max(lever.min_val, min(lever.max_val, new_val))
            mutated[param_name] = new_val

        return mutated

    def _sample_random_params(self) -> Dict[str, Any]:
        """Sample random parameter values (same as RandomSearch)."""
        params = {}

        for name, lever in self.levers.items():
            if lever.data_type == 'categorical':
                params[name] = random.choice(lever.options)
            elif lever.data_type == 'bool':
                params[name] = random.choice([True, False])
            elif lever.data_type == 'int':
                params[name] = random.randint(int(lever.min_val), int(lever.max_val))
            elif lever.data_type == 'float':
                params[name] = random.uniform(lever.min_val, lever.max_val)

        return params


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization.

    Performs rolling window optimization to avoid overfitting.
    - In-sample period: Optimize parameters
    - Out-of-sample period: Test on unseen data
    - Roll forward and repeat

    Critical for robust strategy development.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        objective_function: Callable[[Dict, pd.DataFrame], float],
        levers: Dict[str, FeatureLever],
        optimizer_class: type = RandomSearchOptimizer,
        in_sample_months: int = 12,
        out_sample_months: int = 3,
        maximize: bool = True
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            data: Full dataset with DatetimeIndex
            objective_function: Function(params, data_subset) -> score
            levers: Parameter levers to optimize
            optimizer_class: Which optimizer to use (Grid, Random, GA)
            in_sample_months: Months for training/optimization
            out_sample_months: Months for testing
            maximize: Maximize or minimize objective
        """
        self.data = data
        self.objective_function = objective_function
        self.levers = levers
        self.optimizer_class = optimizer_class
        self.in_sample_months = in_sample_months
        self.out_sample_months = out_sample_months
        self.maximize = maximize

        self.results: List[Dict] = []

    def optimize(
        self,
        optimizer_kwargs: Optional[Dict] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run walk-forward optimization.

        Args:
            optimizer_kwargs: Parameters to pass to optimizer
            verbose: Print progress

        Returns:
            DataFrame with walk-forward results
        """
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        # Get date range
        if isinstance(self.data.index, pd.MultiIndex):
            dates = self.data.index.get_level_values(1).unique()
        else:
            dates = self.data.index

        dates = pd.to_datetime(dates).sort_values()

        start_date = dates.min()
        end_date = dates.max()

        current_date = start_date

        fold = 0
        while current_date + pd.DateOffset(months=self.in_sample_months + self.out_sample_months) <= end_date:
            fold += 1

            # Define periods
            is_start = current_date
            is_end = current_date + pd.DateOffset(months=self.in_sample_months)
            oos_start = is_end
            oos_end = is_end + pd.DateOffset(months=self.out_sample_months)

            if verbose:
                print(f"\n{'='*70}")
                print(f"Walk-Forward Fold {fold}")
                print(f"{'='*70}")
                print(f"In-Sample:  {is_start.date()} to {is_end.date()}")
                print(f"Out-Sample: {oos_start.date()} to {oos_end.date()}")

            # Split data
            is_data = self._filter_data_by_date(self.data, is_start, is_end)
            oos_data = self._filter_data_by_date(self.data, oos_start, oos_end)

            # Optimize on in-sample data
            def is_objective(params):
                return self.objective_function(params, is_data)

            optimizer = self.optimizer_class(
                objective_function=is_objective,
                levers=self.levers,
                maximize=self.maximize
            )

            is_result = optimizer.optimize(**optimizer_kwargs, verbose=False)

            if verbose:
                print(f"In-Sample Best Score: {is_result.best_score:.4f}")

            # Test on out-of-sample data
            oos_score = self.objective_function(is_result.best_config, oos_data)

            if verbose:
                print(f"Out-Sample Score: {oos_score:.4f}")

            # Store results
            self.results.append({
                'fold': fold,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'is_score': is_result.best_score,
                'oos_score': oos_score,
                'best_params': is_result.best_config
            })

            # Roll forward
            current_date = oos_end

        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)

        if verbose:
            print(f"\n{'='*70}")
            print("Walk-Forward Summary")
            print(f"{'='*70}")
            print(f"Average In-Sample Score:  {results_df['is_score'].mean():.4f}")
            print(f"Average Out-Sample Score: {results_df['oos_score'].mean():.4f}")
            print(f"Out-Sample Std:           {results_df['oos_score'].std():.4f}")

        return results_df

    def _filter_data_by_date(
        self,
        data: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Filter data by date range."""
        if isinstance(data.index, pd.MultiIndex):
            # MultiIndex (ticker, date)
            dates = data.index.get_level_values(1)
            mask = (dates >= start_date) & (dates < end_date)
            return data[mask]
        else:
            # DatetimeIndex
            return data.loc[start_date:end_date]


# Convenience functions
def quick_optimize(
    objective_function: Callable,
    levers: Dict[str, FeatureLever],
    method: str = "random",
    n_iter: int = 100,
    maximize: bool = True,
    verbose: bool = True
) -> OptimizationResult:
    """
    Quick optimization with sensible defaults.

    Args:
        objective_function: Function to optimize
        levers: Parameter levers
        method: "grid", "random", or "genetic"
        n_iter: Number of iterations (for random/genetic)
        maximize: Maximize or minimize
        verbose: Print progress

    Returns:
        OptimizationResult
    """
    if method == "grid":
        optimizer = GridSearchOptimizer(objective_function, levers, maximize)
        return optimizer.optimize(verbose=verbose)

    elif method == "random":
        optimizer = RandomSearchOptimizer(objective_function, levers, maximize)
        return optimizer.optimize(n_iter=n_iter, verbose=verbose)

    elif method == "genetic":
        optimizer = GeneticAlgorithmOptimizer(objective_function, levers, maximize)
        return optimizer.optimize(verbose=verbose)

    else:
        raise ValueError(f"Unknown method: {method}")
