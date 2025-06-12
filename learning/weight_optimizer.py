"""
Weight Optimization System for Investigation Agent

This module implements a dynamic weight optimization system for ranking hypotheses
during metric anomaly investigations. It adjusts the importance of different factors
based on their historical effectiveness within specific investigation periods.

Key components:
- Investigation period extraction from natural language
- Dynamic weight adjustment based on performance
- Performance tracking and analysis
- Persistent storage of optimization history
"""

from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from scipy.optimize import minimize
import re

@dataclass
class InvestigationPeriod:
    """
    Represents a time period for investigation with automatic date extraction.
    
    Attributes:
        start_date: Beginning of investigation period
        end_date: End of investigation period
    """
    start_date: datetime
    end_date: datetime
    
    @classmethod
    def from_text(cls, text: str) -> 'InvestigationPeriod':
        """
        Extracts date range from natural language investigation text.
        
        Args:
            text: Investigation description containing date range
                 e.g. "spike in political groups engagement during April to July 2025"
        
        Returns:
            InvestigationPeriod instance with extracted dates
        
        Raises:
            ValueError: If date range cannot be extracted from text
        """
        # Define month names for pattern matching
        months = "January|February|March|April|May|June|July|August|September|October|November|December"
        # Pattern matches: "during/from/between Month to/- Month Year"
        pattern = f"(?:during|from|between)?\\s*({months})\\s+(?:to|-)\\s*({months})\\s+(\\d{{4}})"
        
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            raise ValueError("Could not extract date range from investigation text")
            
        start_month, end_month, year = match.groups()
        
        # Convert start month to datetime (1st day of month)
        start_date = datetime.strptime(f"{start_month} 1 {year}", "%B %d %Y")
        
        # Convert end month to datetime (last day of month)
        end_date = datetime.strptime(f"{end_month} 1 {year}", "%B %d %Y")
        # Calculate last day of end month
        end_date = end_date.replace(day=28) + timedelta(days=4)
        end_date = end_date - timedelta(days=end_date.day)
        
        return cls(start_date=start_date, end_date=end_date)
    
    @property
    def duration_days(self) -> int:
        """Calculate the duration of investigation period in days."""
        return (self.end_date - self.start_date).days

@dataclass
class WeightConfig:
    """
    Configuration for hypothesis ranking weights.
    
    The weights determine the importance of different factors in hypothesis ranking:
    - selection_weight: User feedback importance (50%)
    - execution_weight: Technical success importance (20%)
    - context_weight: Scenario match importance (20%)
    - temporal_weight: Time relevance importance (5%)
    - relationship_weight: Inter-hypothesis connection importance (5%)
    
    All weights must sum to 1.0 (100%)
    """
    selection_weight: float = 0.5      # Increased to 50%
    execution_weight: float = 0.2      # Kept at 20%
    context_weight: float = 0.2        # Kept at 20%
    temporal_weight: float = 0.05      # Reduced to 5%
    relationship_weight: float = 0.05  # Reduced to 5%
    
    def __post_init__(self):
        """Validates that weights sum to 1.0"""
        total = sum([
            self.selection_weight,
            self.execution_weight,
            self.context_weight,
            self.temporal_weight,
            self.relationship_weight
        ])
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

@dataclass
class PerformanceMetrics:
    """
    Stores performance metrics for a single hypothesis investigation.
    
    Attributes:
        hypothesis_id: Unique identifier for the hypothesis
        user_rating: User-provided rating of hypothesis quality (0-1)
        execution_success: Whether hypothesis testing completed successfully
        context_match_score: How well hypothesis matches investigation context (0-1)
        temporal_relevance: Time-based relevance score (0-1)
        relationship_strength: Connection strength with other hypotheses (0-1)
        final_success: Whether hypothesis helped explain the anomaly
        timestamp: When these metrics were recorded (ISO format)
    """
    hypothesis_id: str
    user_rating: float
    execution_success: bool
    context_match_score: float
    temporal_relevance: float
    relationship_strength: float
    final_success: bool
    timestamp: str  # ISO format datetime string

class WeightOptimizer:
    """
    Optimizes hypothesis ranking weights based on historical performance.
    
    This class manages the dynamic adjustment of weights used to rank hypotheses
    during investigations. It tracks performance metrics, optimizes weights based
    on success patterns, and maintains a history of weight configurations.
    """
    
    def __init__(self, 
                 investigation_text: str,
                 initial_config: Optional[WeightConfig] = None,
                 learning_rate: float = 0.1):
        """
        Initialize the weight optimizer.
        
        Args:
            investigation_text: Description of investigation including date range
            initial_config: Optional starting weight configuration
            learning_rate: Rate of weight adjustment (0-1)
        """
        # Extract investigation period from description
        self.investigation_period = InvestigationPeriod.from_text(investigation_text)
        
        # Initialize weights (use provided config or defaults)
        self.current_config = initial_config or WeightConfig()
        self.learning_rate = learning_rate
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_history: List[WeightConfig] = [self.current_config]
        
        self.logger = logging.getLogger(__name__)
        
        # Load any existing historical data
        self._load_history()

    def record_performance(self, metrics: PerformanceMetrics) -> None:
        """
        Record new performance metrics and trigger weight optimization.
        
        Args:
            metrics: Performance metrics for a hypothesis investigation
        """
        self.performance_history.append(metrics)
        self._save_history()
        
        # Only optimize if we have enough data points
        if len(self._get_investigation_period_history()) >= 10:
            self._optimize_weights()

    def get_current_weights(self) -> WeightConfig:
        """Get the current weight configuration."""
        return self.current_config

    def _optimize_weights(self) -> None:
        """
        Optimize weights based on historical performance.
        
        Uses scipy.optimize.minimize to find weights that maximize the correlation
        between weighted scores and final success outcomes. The optimization is
        constrained to ensure weights sum to 1.0 and each weight is between 0.1
        and 0.5.
        """
        period_history = self._get_investigation_period_history()
        if not period_history:
            return

        # Convert metrics to numpy arrays for optimization
        X = np.array([
            [
                m.user_rating,
                float(m.execution_success),
                m.context_match_score,
                m.temporal_relevance,
                m.relationship_strength
            ]
            for m in period_history
        ])
        
        y = np.array([float(m.final_success) for m in period_history])
        
        # Objective function minimizes negative MSE (maximizes accuracy)
        def objective(weights):
            predictions = np.dot(X, weights)
            return -np.mean((predictions - y) ** 2)
        
        # Constraints ensure valid weight distribution
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum to 1
            {'type': 'ineq', 'fun': lambda w: w - 0.1}       # Minimum 0.1 each
        )
        
        # Current weights as starting point
        initial_weights = np.array([
            self.current_config.selection_weight,
            self.current_config.execution_weight,
            self.current_config.context_weight,
            self.current_config.temporal_weight,
            self.current_config.relationship_weight
        ])
        
        # Optimize weights
        result = minimize(
            objective,
            initial_weights,
            constraints=constraints,
            bounds=[(0.1, 0.5) for _ in range(5)]
        )
        
        if result.success:
            # Apply gradual update using learning rate
            new_weights = (1 - self.learning_rate) * initial_weights + \
                        self.learning_rate * result.x
            
            # Update current configuration
            self.current_config = WeightConfig(
                selection_weight=new_weights[0],
                execution_weight=new_weights[1],
                context_weight=new_weights[2],
                temporal_weight=new_weights[3],
                relationship_weight=new_weights[4]
            )
            
            self.optimization_history.append(self.current_config)
            self._save_history()
            
            self.logger.info(f"Updated weights: {self.current_config}")
        else:
            self.logger.warning("Weight optimization failed")

    def _get_investigation_period_history(self) -> List[PerformanceMetrics]:
        """
        Get performance metrics within the investigation period.
        
        Returns:
            List of performance metrics recorded between start_date and end_date
        """
        return [
            metric for metric in self.performance_history
            if self.investigation_period.start_date <= 
               datetime.fromisoformat(metric.timestamp) <= 
               self.investigation_period.end_date
        ]

    def analyze_weight_effectiveness(self) -> Dict:
        """
        Analyze the effectiveness of current weights.
        
        Returns:
            Dictionary containing:
            - Correlations between each factor and final success
            - Overall success rate
            - Sample size
            - Current weights
            - Number of optimizations performed
            - Investigation period details
        """
        period_history = self._get_investigation_period_history()
        if not period_history:
            return {
                "message": "No data available for the investigation period",
                "period": {
                    "start": self.investigation_period.start_date.isoformat(),
                    "end": self.investigation_period.end_date.isoformat(),
                    "duration_days": self.investigation_period.duration_days
                }
            }
        
        # Calculate correlations between factors and success
        correlations = {
            "selection": self._calculate_correlation(
                [m.user_rating for m in period_history],
                [m.final_success for m in period_history]
            ),
            "execution": self._calculate_correlation(
                [float(m.execution_success) for m in period_history],
                [m.final_success for m in period_history]
            ),
            "context": self._calculate_correlation(
                [m.context_match_score for m in period_history],
                [m.final_success for m in period_history]
            ),
            "temporal": self._calculate_correlation(
                [m.temporal_relevance for m in period_history],
                [m.final_success for m in period_history]
            ),
            "relationship": self._calculate_correlation(
                [m.relationship_strength for m in period_history],
                [m.final_success for m in period_history]
            )
        }
        
        # Calculate overall success rate
        success_rate = np.mean([m.final_success for m in period_history])
        
        return {
            "correlations": correlations,
            "success_rate": success_rate,
            "sample_size": len(period_history),
            "current_weights": self.current_config.__dict__,
            "optimization_count": len(self.optimization_history),
            "period": {
                "start": self.investigation_period.start_date.isoformat(),
                "end": self.investigation_period.end_date.isoformat(),
                "duration_days": self.investigation_period.duration_days
            }
        }

    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate correlation coefficient between two metrics.
        
        Args:
            x: First metric values
            y: Second metric values
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if not x or not y:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def get_weight_history(self) -> List[Dict]:
        """Get the history of weight configurations."""
        return [config.__dict__ for config in self.optimization_history]

    def _save_history(self) -> None:
        """
        Save performance and optimization history to disk.
        
        Creates a learning_data directory and saves:
        - performance_history.json: All recorded performance metrics
        - weight_optimization.json: History of weight configurations
        """
        data_path = Path("learning_data")
        data_path.mkdir(exist_ok=True)
        
        # Save performance metrics
        with open(data_path / "performance_history.json", 'w') as f:
            json.dump([vars(m) for m in self.performance_history], f)
        
        # Save weight optimization history
        with open(data_path / "weight_optimization.json", 'w') as f:
            json.dump([vars(w) for w in self.optimization_history], f)

    def _load_history(self) -> None:
        """
        Load historical data from disk.
        
        Loads previously saved performance metrics and weight configurations
        if they exist.
        """
        data_path = Path("learning_data")
        
        # Load performance history
        perf_path = data_path / "performance_history.json"
        if perf_path.exists():
            with open(perf_path, 'r') as f:
                data = json.load(f)
                self.performance_history = [PerformanceMetrics(**m) for m in data]
        
        # Load optimization history
        opt_path = data_path / "weight_optimization.json"
        if opt_path.exists():
            with open(opt_path, 'r') as f:
                data = json.load(f)
                self.optimization_history = [WeightConfig(**w) for w in data]
                if self.optimization_history:
                    self.current_config = self.optimization_history[-1] 