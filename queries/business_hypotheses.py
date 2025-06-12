from typing import Dict, Any
from jinja2 import Template
from .base_queries import QueryTemplate

# Political Engagement Patterns
POLITICAL_ENGAGEMENT = QueryTemplate("""
    WITH election_weeks AS (
        SELECT 
            date_trunc('week', timestamp) as week_start,
            COUNT(*) as election_events
        FROM events
        WHERE event_type = 'election'
        GROUP BY 1
    ),
    weekly_engagement AS (
        SELECT 
            date_trunc('week', h.timestamp) as week_start,
            g.group_name,
            AVG(h.comments_made) as avg_comments,
            AVG(h.reactions_given) as avg_reactions,
            COUNT(DISTINCT h.city_id) as active_cities
        FROM hourly_active_users h
        JOIN content_groups g ON h.group_id = g.group_id
        WHERE g.group_name = 'Politics'
        GROUP BY 1, 2
    )
    SELECT 
        w.week_start,
        w.group_name,
        w.avg_comments,
        w.avg_reactions,
        w.active_cities,
        COALESCE(e.election_events, 0) as election_events,
        w.avg_comments / AVG(w.avg_comments) OVER (
            ORDER BY w.week_start
            ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) as comment_ratio
    FROM weekly_engagement w
    LEFT JOIN election_weeks e ON w.week_start = e.week_start
    WHERE w.week_start BETWEEN '{{ start_date }}' AND '{{ end_date }}'
    ORDER BY w.week_start
""")

# Content Virality Analysis
VIRAL_CONTENT_PATTERNS = QueryTemplate("""
    WITH content_metrics AS (
        SELECT 
            timestamp,
            group_id,
            content_shared,
            reactions_given,
            comments_made,
            LAG(content_shared, 24) OVER (
                PARTITION BY group_id 
                ORDER BY timestamp
            ) as prev_day_shares
        FROM hourly_active_users
        WHERE timestamp BETWEEN '{{ start_date }}' AND '{{ end_date }}'
        {% if group_id %}
        AND group_id = {{ group_id }}
        {% endif %}
    ),
    viral_periods AS (
        SELECT 
            timestamp,
            group_id,
            content_shared,
            CASE 
                WHEN content_shared > 2 * prev_day_shares 
                AND content_shared > {{ min_shares }} THEN 1 
                ELSE 0 
            END as is_viral
        FROM content_metrics
    )
    SELECT 
        date_trunc('hour', timestamp) as hour,
        g.group_name,
        AVG(content_shared) as avg_shares,
        SUM(is_viral) as viral_events,
        AVG(CASE WHEN is_viral = 1 THEN content_shared ELSE NULL END) as viral_shares
    FROM viral_periods v
    JOIN content_groups g ON v.group_id = g.group_id
    GROUP BY 1, 2
    HAVING SUM(is_viral) > 0
    ORDER BY 1 DESC
""")

# Cross-Group Influence
CROSS_GROUP_INFLUENCE = QueryTemplate("""
    WITH group_activity AS (
        SELECT 
            timestamp,
            g1.group_name as source_group,
            g2.group_name as target_group,
            h1.active_users as source_users,
            h2.active_users as target_users,
            h2.content_shared as target_shares
        FROM hourly_active_users h1
        JOIN content_groups g1 ON h1.group_id = g1.group_id
        CROSS JOIN LATERAL (
            SELECT h2.active_users, h2.content_shared, g2.group_name
            FROM hourly_active_users h2
            JOIN content_groups g2 ON h2.group_id = g2.group_id
            WHERE h2.timestamp = h1.timestamp
            AND g2.group_id != g1.group_id
        ) h2
        WHERE h1.timestamp BETWEEN '{{ start_date }}' AND '{{ end_date }}'
    )
    SELECT 
        source_group,
        target_group,
        CORR(source_users, target_shares) as influence_score,
        COUNT(*) as sample_size,
        AVG(target_shares) as avg_target_shares
    FROM group_activity
    GROUP BY 1, 2
    HAVING COUNT(*) >= {{ min_samples }}
    ORDER BY influence_score DESC
""")

# User Engagement Patterns by Time
TIME_BASED_PATTERNS = QueryTemplate("""
    WITH hourly_patterns AS (
        SELECT 
            EXTRACT(DOW FROM timestamp) as day_of_week,
            EXTRACT(HOUR FROM timestamp) as hour_of_day,
            g.group_name,
            AVG(h.active_users) as avg_users,
            AVG(h.comments_made) as avg_comments,
            AVG(h.content_shared) as avg_shares
        FROM hourly_active_users h
        JOIN content_groups g ON h.group_id = g.group_id
        WHERE timestamp BETWEEN '{{ start_date }}' AND '{{ end_date }}'
        {% if group_id %}
        AND h.group_id = {{ group_id }}
        {% endif %}
        GROUP BY 1, 2, 3
    )
    SELECT 
        group_name,
        day_of_week,
        hour_of_day,
        avg_users,
        avg_comments,
        avg_shares,
        avg_users / AVG(avg_users) OVER (
            PARTITION BY group_name
        ) as relative_activity
    FROM hourly_patterns
    ORDER BY 
        group_name,
        relative_activity DESC
""")

# Geographic Trend Analysis
GEOGRAPHIC_TRENDS = QueryTemplate("""
    WITH city_metrics AS (
        SELECT 
            h.timestamp,
            g.group_name,
            gh.city_name,
            gh.region,
            gh.country,
            SUM(h.active_users) as total_users,
            SUM(h.comments_made) as total_comments,
            SUM(h.content_shared) as total_shares
        FROM hourly_active_users h
        JOIN content_groups g ON h.group_id = g.group_id
        JOIN geo_hierarchy gh ON h.city_id = gh.city_id
        WHERE h.timestamp BETWEEN '{{ start_date }}' AND '{{ end_date }}'
        GROUP BY 1, 2, 3, 4, 5
    ),
    regional_stats AS (
        SELECT 
            group_name,
            region,
            country,
            AVG(total_users) as avg_users,
            AVG(total_comments) as avg_comments,
            AVG(total_shares) as avg_shares,
            STDDEV(total_users) as user_stddev
        FROM city_metrics
        GROUP BY 1, 2, 3
    )
    SELECT 
        group_name,
        region,
        country,
        avg_users,
        avg_comments,
        avg_shares,
        avg_comments / NULLIF(avg_users, 0) as engagement_rate,
        user_stddev / NULLIF(avg_users, 0) as user_variability
    FROM regional_stats
    WHERE avg_users > {{ min_users }}
    ORDER BY engagement_rate DESC
""") 