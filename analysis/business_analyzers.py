from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .base_analyzers import BaseAnalyzer, AnalysisResult

class PoliticalEngagementAnalyzer(BaseAnalyzer):
    """Analyzes political engagement during election periods"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Calculate average engagement metrics
        election_weeks = data[data['election_events'] > 0]
        normal_weeks = data[data['election_events'] == 0]
        
        election_stats = {
            'avg_comments': election_weeks['avg_comments'].mean(),
            'avg_reactions': election_weeks['avg_reactions'].mean(),
            'active_cities': election_weeks['active_cities'].mean()
        }
        
        normal_stats = {
            'avg_comments': normal_weeks['avg_comments'].mean(),
            'avg_reactions': normal_weeks['avg_reactions'].mean(),
            'active_cities': normal_weeks['active_cities'].mean()
        }
        
        # Calculate percentage increases
        increases = {
            metric: ((election_stats[metric] - normal_stats[metric]) / 
                    normal_stats[metric] * 100)
            for metric in election_stats.keys()
        }
        
        # Determine significance
        is_significant = any(abs(inc) > self.config.get('significance_threshold', 20) 
                           for inc in increases.values())
        
        # Generate findings
        findings = []
        if increases['avg_comments'] > 0:
            findings.append(f"Comments increase by {increases['avg_comments']:.1f}% during election weeks")
        if increases['avg_reactions'] > 0:
            findings.append(f"Reactions increase by {increases['avg_reactions']:.1f}% during election weeks")
        if increases['active_cities'] > 0:
            findings.append(f"Geographic engagement spreads to {increases['active_cities']:.1f}% more cities")
        
        return AnalysisResult(
            is_anomaly=is_significant,
            confidence=min(1.0, max(abs(inc) for inc in increases.values()) / 100),
            details={
                'election_stats': election_stats,
                'normal_stats': normal_stats,
                'increases': increases
            },
            suggested_actions=[
                "Monitor engagement levels in highly active cities",
                "Analyze content themes during high-engagement periods",
                "Track cross-group spillover effects"
            ]
        )

class ViralContentAnalyzer(BaseAnalyzer):
    """Analyzes patterns in viral content"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Identify viral events
        viral_events = data[data['viral_events'] > 0]
        
        if len(viral_events) == 0:
            return AnalysisResult(
                is_anomaly=False,
                confidence=1.0,
                details={'message': 'No viral events detected'},
                suggested_actions=['Lower viral threshold for more events']
            )
        
        # Analyze temporal patterns
        viral_events['hour'] = pd.to_datetime(viral_events['hour'])
        peak_hours = viral_events.groupby(viral_events['hour'].dt.hour)['viral_events'].sum()
        peak_hour = peak_hours.idxmax()
        
        # Calculate viral characteristics
        details = {
            'total_viral_events': viral_events['viral_events'].sum(),
            'avg_viral_shares': viral_events['viral_shares'].mean(),
            'peak_hour': peak_hour,
            'viral_share_ratio': viral_events['viral_shares'].mean() / data['avg_shares'].mean()
        }
        
        # Generate insights
        actions = [
            f"Focus content releases around {peak_hour}:00",
            "Analyze common characteristics of viral content",
            "Monitor engagement velocity for early viral detection"
        ]
        
        return AnalysisResult(
            is_anomaly=True,  # Viral events are always noteworthy
            confidence=min(1.0, details['viral_share_ratio'] / 10),
            details=details,
            suggested_actions=actions
        )

class CrossGroupAnalyzer(BaseAnalyzer):
    """Analyzes influence between different groups"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Find strongest correlations
        significant_correlations = data[
            abs(data['influence_score']) > self.config.get('correlation_threshold', 0.5)
        ]
        
        if len(significant_correlations) == 0:
            return AnalysisResult(
                is_anomaly=False,
                confidence=1.0,
                details={'message': 'No significant cross-group influence detected'},
                suggested_actions=['Analyze shorter time periods for temporal effects']
            )
        
        # Identify key relationships
        top_influencers = significant_correlations.groupby('source_group')[
            'influence_score'
        ].mean().sort_values(ascending=False)
        
        details = {
            'top_influencer': top_influencers.index[0],
            'max_correlation': significant_correlations['influence_score'].max(),
            'significant_pairs': len(significant_correlations),
            'influence_network': significant_correlations.to_dict('records')
        }
        
        # Generate recommendations
        actions = [
            f"Monitor {details['top_influencer']} as key influence source",
            "Analyze content crossover between correlated groups",
            "Consider cross-promotion strategies for highly correlated groups"
        ]
        
        return AnalysisResult(
            is_anomaly=len(significant_correlations) > self.config.get('min_significant_pairs', 3),
            confidence=min(1.0, abs(details['max_correlation'])),
            details=details,
            suggested_actions=actions
        )

class TimePatternAnalyzer(BaseAnalyzer):
    """Analyzes temporal engagement patterns"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Find peak engagement times
        peak_periods = data.nlargest(5, 'relative_activity')
        
        # Map day numbers to names
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        peak_periods['day_name'] = peak_periods['day_of_week'].map(lambda x: days[int(x)])
        
        details = {
            'peak_day': peak_periods.iloc[0]['day_name'],
            'peak_hour': int(peak_periods.iloc[0]['hour_of_day']),
            'max_relative_activity': peak_periods.iloc[0]['relative_activity'],
            'top_periods': peak_periods[['day_name', 'hour_of_day', 'relative_activity']].to_dict('records')
        }
        
        # Generate time-based recommendations
        actions = [
            f"Schedule key content for {details['peak_day']} at {details['peak_hour']}:00",
            "Adjust content strategy for peak engagement windows",
            "Consider time zone differences for global audience"
        ]
        
        return AnalysisResult(
            is_anomaly=details['max_relative_activity'] > self.config.get('activity_threshold', 1.5),
            confidence=min(1.0, details['max_relative_activity'] / 2),
            details=details,
            suggested_actions=actions
        )

class GeographicAnalyzer(BaseAnalyzer):
    """Analyzes geographic engagement patterns"""
    
    def analyze(self, data: pd.DataFrame) -> AnalysisResult:
        # Find regions with highest engagement
        top_regions = data.nlargest(5, 'engagement_rate')
        
        details = {
            'top_region': {
                'name': top_regions.iloc[0]['region'],
                'country': top_regions.iloc[0]['country'],
                'engagement_rate': top_regions.iloc[0]['engagement_rate']
            },
            'most_consistent': data.nsmallest(1, 'user_variability').iloc[0]['region'],
            'regional_stats': top_regions.to_dict('records')
        }
        
        # Calculate regional variations
        avg_engagement = data['engagement_rate'].mean()
        engagement_std = data['engagement_rate'].std()
        has_significant_variation = (
            details['top_region']['engagement_rate'] > 
            avg_engagement + 2 * engagement_std
        )
        
        actions = [
            f"Focus growth efforts in {details['top_region']['region']}",
            f"Analyze success factors in {details['top_region']['region']}",
            "Consider regional content customization"
        ]
        
        if has_significant_variation:
            actions.append("Investigate causes of regional engagement disparities")
        
        return AnalysisResult(
            is_anomaly=has_significant_variation,
            confidence=min(1.0, details['top_region']['engagement_rate'] / 
                         (avg_engagement + engagement_std)),
            details=details,
            suggested_actions=actions
        ) 