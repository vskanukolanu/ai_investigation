from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class BusinessHypothesis:
    """Business-oriented hypothesis template"""
    name: str
    description: str
    query_name: str
    required_metrics: List[str]
    default_params: Dict
    validation_rules: Optional[Dict] = None
    explanation_template: str = ""

# Political Engagement Hypotheses
ELECTION_ENGAGEMENT = BusinessHypothesis(
    name="election_week_engagement",
    description="Analyze if political engagement increases during election weeks",
    query_name="POLITICAL_ENGAGEMENT",
    required_metrics=["comments_made", "reactions_given"],
    default_params={
        "start_date": (datetime.now() - timedelta(days=90)).isoformat(),
        "end_date": datetime.now().isoformat()
    },
    explanation_template="""
    During the period from {start_date} to {end_date}:
    - Average comments during election weeks: {election_week_comments}
    - Average comments during normal weeks: {normal_week_comments}
    - Engagement increase: {engagement_increase}%
    
    Key findings:
    {findings}
    """
)

VIRAL_CONTENT = BusinessHypothesis(
    name="viral_content_analysis",
    description="Identify patterns in content that goes viral within groups",
    query_name="VIRAL_CONTENT_PATTERNS",
    required_metrics=["content_shared", "reactions_given"],
    default_params={
        "min_shares": 100,
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat()
    },
    explanation_template="""
    Viral content analysis for {group_name}:
    - Number of viral events: {viral_events}
    - Average shares during viral events: {viral_shares}
    - Peak sharing time: {peak_time}
    
    Characteristics:
    {characteristics}
    """
)

CROSS_GROUP_ENGAGEMENT = BusinessHypothesis(
    name="cross_group_influence",
    description="Analyze how activity in one group affects engagement in others",
    query_name="CROSS_GROUP_INFLUENCE",
    required_metrics=["active_users", "content_shared"],
    default_params={
        "min_samples": 168,  # 1 week of hourly data
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat()
    },
    explanation_template="""
    Cross-group influence analysis:
    - Most influential group: {top_influencer}
    - Most responsive group: {top_responder}
    - Strongest correlation: {max_correlation}
    
    Key relationships:
    {relationships}
    """
)

TIME_PATTERNS = BusinessHypothesis(
    name="engagement_time_patterns",
    description="Identify optimal engagement times for different groups",
    query_name="TIME_BASED_PATTERNS",
    required_metrics=["active_users", "comments_made", "content_shared"],
    default_params={
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat()
    },
    explanation_template="""
    Time-based engagement patterns for {group_name}:
    - Peak activity day: {peak_day}
    - Peak activity hour: {peak_hour}
    - Relative engagement: {relative_engagement}x average
    
    Recommended posting times:
    {recommendations}
    """
)

GEOGRAPHIC_ENGAGEMENT = BusinessHypothesis(
    name="geographic_trends",
    description="Analyze regional differences in group engagement",
    query_name="GEOGRAPHIC_TRENDS",
    required_metrics=["active_users", "comments_made", "content_shared"],
    default_params={
        "min_users": 100,
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat()
    },
    explanation_template="""
    Geographic engagement analysis for {group_name}:
    - Most engaged region: {top_region}
    - Highest engagement rate: {max_engagement}
    - Most consistent region: {most_consistent}
    
    Regional insights:
    {insights}
    """
)

# Dictionary of all business hypotheses
BUSINESS_HYPOTHESES = {
    "election_engagement": ELECTION_ENGAGEMENT,
    "viral_content": VIRAL_CONTENT,
    "cross_group": CROSS_GROUP_ENGAGEMENT,
    "time_patterns": TIME_PATTERNS,
    "geographic": GEOGRAPHIC_ENGAGEMENT
} 