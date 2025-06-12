from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    is_anomaly: bool
    confidence: float
    details: Dict
    suggested_actions: List[str]

class BaseAnalyzer:
    """Base class for analysis scripts"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        raise NotImplementedError("Subclasses must implement analyze method")

class TimeSeriesAnalyzer(BaseAnalyzer):
    """Analyzes time series data for anomalies"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Ensure required columns exist
        required_cols = ['timestamp', 'metric_value', 'expected_value']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        # Calculate z-scores
        z_scores = np.abs((data['metric_value'] - data['expected_value']) / 
                         data['metric_value'].std())
        
        # Identify anomalies
        threshold = self.config.get('z_score_threshold', 3.0)
        anomalies = z_scores > threshold
        
        # Calculate confidence based on magnitude and duration
        confidence = min(1.0, np.mean(z_scores[anomalies]) / 10.0) if any(anomalies) else 0.0
        
        # Generate insights
        details = {
            'total_points': len(data),
            'anomaly_points': sum(anomalies),
            'max_deviation': z_scores.max(),
            'anomaly_timestamps': data.loc[anomalies, 'timestamp'].tolist()
        }
        
        # Suggest actions based on findings
        actions = []
        if any(anomalies):
            actions.append(f"Investigate {sum(anomalies)} points with z-score > {threshold}")
            if len(data.loc[anomalies, 'timestamp'].unique()) > 1:
                actions.append("Check for systemic issues during anomaly period")
        
        return AnalysisResult(
            is_anomaly=any(anomalies),
            confidence=confidence,
            details=details,
            suggested_actions=actions
        )

class SegmentAnalyzer(BaseAnalyzer):
    """Analyzes differences between segments"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Perform one-way ANOVA to test for significant differences
        segments = data['segment'].unique()
        segment_groups = [data[data['segment'] == seg]['value'] for seg in segments]
        
        f_stat, p_value = stats.f_oneway(*segment_groups)
        
        # Calculate effect size (eta-squared)
        groups_mean = np.array([group.mean() for group in segment_groups])
        grand_mean = data['value'].mean()
        n = len(data)
        
        ss_between = sum(len(group) * (group_mean - grand_mean) ** 2 
                        for group, group_mean in zip(segment_groups, groups_mean))
        ss_total = sum((data['value'] - grand_mean) ** 2)
        eta_squared = ss_between / ss_total
        
        # Determine if there are significant differences
        is_significant = p_value < self.config.get('alpha', 0.05)
        
        details = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'segment_means': {seg: data[data['segment'] == seg]['value'].mean() 
                            for seg in segments},
            'segment_sizes': {seg: len(data[data['segment'] == seg]) 
                            for seg in segments}
        }
        
        actions = []
        if is_significant:
            # Find segments with largest differences
            segment_means = pd.Series(details['segment_means'])
            max_diff_segments = segment_means.nlargest(2).index
            actions.append(f"Investigate difference between {max_diff_segments[0]} "
                         f"and {max_diff_segments[1]}")
            
            # Check for small segment sizes
            small_segments = [seg for seg, size in details['segment_sizes'].items() 
                            if size < self.config.get('min_segment_size', 30)]
            if small_segments:
                actions.append(f"Warning: Small sample size for segments: {small_segments}")
        
        return AnalysisResult(
            is_anomaly=is_significant,
            confidence=1 - p_value if is_significant else 0.0,
            details=details,
            suggested_actions=actions
        )

class CorrelationAnalyzer(BaseAnalyzer):
    """Analyzes correlations between metrics"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Find significant correlations
        threshold = self.config.get('correlation_threshold', 0.7)
        significant_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    significant_corr.append({
                        'metric1': corr_matrix.columns[i],
                        'metric2': corr_matrix.columns[j],
                        'correlation': corr
                    })
        
        details = {
            'correlation_matrix': corr_matrix.to_dict(),
            'significant_correlations': significant_corr,
            'sample_size': len(data)
        }
        
        actions = []
        for corr in significant_corr:
            direction = 'positive' if corr['correlation'] > 0 else 'negative'
            actions.append(f"Investigate {direction} correlation between "
                         f"{corr['metric1']} and {corr['metric2']}")
        
        return AnalysisResult(
            is_anomaly=len(significant_corr) > 0,
            confidence=max([abs(c['correlation']) for c in significant_corr], default=0.0),
            details=details,
            suggested_actions=actions
        ) 