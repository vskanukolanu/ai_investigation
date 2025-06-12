import duckdb
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Create a connection to DuckDB
conn = duckdb.connect('investigation.db')

# Create tables for our investigation system

# 1. Location hierarchy for better geographical analysis
conn.execute("""
CREATE TABLE geo_hierarchy (
    city_id INTEGER PRIMARY KEY,
    city_name VARCHAR,
    region VARCHAR,
    country VARCHAR,
    timezone VARCHAR
);
""")

# 2. Content groups/categories
conn.execute("""
CREATE TABLE content_groups (
    group_id INTEGER PRIMARY KEY,
    group_name VARCHAR,
    description VARCHAR,
    parent_group VARCHAR  -- For hierarchical categorization if needed
);
""")

# 3. Hourly active users by location and content group
conn.execute("""
CREATE TABLE hourly_active_users (
    timestamp TIMESTAMP,
    city_id INTEGER,
    group_id INTEGER,
    active_users INTEGER,
    new_users INTEGER,
    returning_users INTEGER,
    FOREIGN KEY (city_id) REFERENCES geo_hierarchy(city_id),
    FOREIGN KEY (group_id) REFERENCES content_groups(group_id)
);
""")

# 4. User engagement metrics by group
conn.execute("""
CREATE TABLE user_engagement (
    timestamp TIMESTAMP,
    city_id INTEGER,
    group_id INTEGER,
    engagement_group VARCHAR,  -- 'highly_engaged', 'moderately_engaged', 'low_engaged', 'dormant'
    metric_type VARCHAR,      -- Different types of engagement metrics
    metric_value DOUBLE,
    FOREIGN KEY (city_id) REFERENCES geo_hierarchy(city_id),
    FOREIGN KEY (group_id) REFERENCES content_groups(group_id)
);
""")

# 5. Engagement metrics definitions and thresholds
conn.execute("""
CREATE TABLE engagement_metrics (
    metric_name VARCHAR PRIMARY KEY,
    description VARCHAR,
    measurement_unit VARCHAR,
    threshold_high DOUBLE,
    threshold_medium DOUBLE,
    threshold_low DOUBLE
);
""")

def generate_sample_data():
    # Generate sample cities
    cities_data = [
        (1, 'New York', 'Northeast', 'US', 'America/New_York', 40.7128, -74.0060),
        (2, 'San Francisco', 'West', 'US', 'America/Los_Angeles', 37.7749, -122.4194),
        (3, 'London', 'Greater London', 'UK', 'Europe/London', 51.5074, -0.1278),
        (4, 'Tokyo', 'Kanto', 'JP', 'Asia/Tokyo', 35.6762, 139.6503),
        (5, 'Singapore', 'Central', 'SG', 'Asia/Singapore', 1.3521, 103.8198)
    ]

    # Generate content groups
    content_groups_data = [
        (1, 'Sports', 'Sports-related content and discussions', None),
        (2, 'Politics', 'Political discussions and news', None),
        (3, 'Arts', 'Visual arts, exhibitions, and artistic discussions', None),
        (4, 'Movies', 'Film discussions, reviews, and news', None),
        (5, 'Music', 'Music discussions, sharing, and news', None)
    ]
    
    # Generate 7 days of hourly data
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(24*7)]
    
    # Generate hourly active users data with realistic patterns
    hourly_users_data = []
    for city_id, _, _, _, timezone, _, _ in cities_data:
        for group_id, group_name, _, _ in content_groups_data:
            # Different base users for different groups
            if group_name == 'Sports':
                base_users = np.random.randint(2000, 4000)
            elif group_name == 'Politics':
                base_users = np.random.randint(1500, 3000)
            elif group_name == 'Music':
                base_users = np.random.randint(2500, 4500)
            elif group_name == 'Movies':
                base_users = np.random.randint(1800, 3500)
            else:  # Arts
                base_users = np.random.randint(1000, 2000)

            for dt in dates:
                # Create daily and weekly patterns
                hour = dt.hour
                day = dt.weekday()
                
                # Adjust patterns based on content type
                if group_name == 'Sports':
                    # More activity during evenings and weekends
                    time_multiplier = 1.8 if 18 <= hour <= 23 else 0.7
                    day_multiplier = 1.5 if day >= 5 else 1.0
                elif group_name == 'Politics':
                    # More activity during working hours on weekdays
                    time_multiplier = 1.5 if 9 <= hour <= 17 else 0.8
                    day_multiplier = 1.3 if day < 5 else 0.7
                elif group_name == 'Music':
                    # Steady throughout day, peak in evening
                    time_multiplier = 1.4 if 16 <= hour <= 23 else 1.0
                    day_multiplier = 1.1
                elif group_name == 'Movies':
                    # Peak in evenings
                    time_multiplier = 1.6 if 19 <= hour <= 23 else 0.9
                    day_multiplier = 1.2 if day >= 5 else 1.0
                else:  # Arts
                    # More activity during daytime
                    time_multiplier = 1.3 if 10 <= hour <= 18 else 0.8
                    day_multiplier = 1.0
                
                active = int(base_users * time_multiplier * day_multiplier * (1 + np.random.normal(0, 0.1)))
                new = int(active * np.random.uniform(0.05, 0.15))
                returning = active - new
                
                hourly_users_data.append({
                    'timestamp': dt,
                    'city_id': city_id,
                    'group_id': group_id,
                    'active_users': active,
                    'new_users': new,
                    'returning_users': returning
                })

    # Generate engagement metrics
    engagement_metrics_data = [
        ('posts_created', 'Number of posts created', 'count', 10.0, 5.0, 1.0),
        ('comments_made', 'Number of comments on posts', 'count', 20.0, 10.0, 3.0),
        ('time_spent', 'Minutes spent in group', 'minutes', 45.0, 20.0, 5.0),
        ('content_shared', 'Items shared with others', 'count', 8.0, 4.0, 1.0),
        ('reactions_given', 'Number of reactions to content', 'count', 30.0, 15.0, 5.0)
    ]

    # Generate user engagement data
    engagement_groups = ['highly_engaged', 'moderately_engaged', 'low_engaged', 'dormant']
    user_engagement_data = []
    
    for dt in dates:
        for city_id, _, _, _, _, _, _ in cities_data:
            for group_id, group_name, _, _ in content_groups_data:
                for metric_name, _, _, high, med, low in engagement_metrics_data:
                    for group in engagement_groups:
                        # Base value depends on engagement group and content type
                        group_multiplier = 1.0
                        if group_name == 'Sports' and metric_name in ['comments_made', 'reactions_given']:
                            group_multiplier = 1.3  # Sports fans are more interactive
                        elif group_name == 'Politics' and metric_name == 'comments_made':
                            group_multiplier = 1.5  # Political discussions generate more comments
                        elif group_name == 'Music' and metric_name == 'content_shared':
                            group_multiplier = 1.4  # Music content is shared more
                        
                        if group == 'highly_engaged':
                            base_value = high * group_multiplier * (1 + np.random.normal(0, 0.1))
                        elif group == 'moderately_engaged':
                            base_value = med * group_multiplier * (1 + np.random.normal(0, 0.1))
                        elif group == 'low_engaged':
                            base_value = low * group_multiplier * (1 + np.random.normal(0, 0.1))
                        else:  # dormant
                            base_value = low * 0.3 * group_multiplier * (1 + np.random.normal(0, 0.1))

                        user_engagement_data.append({
                            'timestamp': dt,
                            'city_id': city_id,
                            'group_id': group_id,
                            'engagement_group': group,
                            'metric_type': metric_name,
                            'metric_value': max(0, base_value)
                        })

    return pd.DataFrame(cities_data, columns=['city_id', 'city_name', 'region', 'country', 'timezone', 'latitude', 'longitude']), \
           pd.DataFrame(content_groups_data, columns=['group_id', 'group_name', 'description', 'parent_group']), \
           pd.DataFrame(hourly_users_data), \
           pd.DataFrame(engagement_metrics_data, columns=['metric_name', 'description', 'measurement_unit', 'threshold_high', 'threshold_medium', 'threshold_low']), \
           pd.DataFrame(user_engagement_data)

# Generate and insert sample data
geo_df, groups_df, hourly_df, metrics_df, engagement_df = generate_sample_data()

conn.execute("INSERT INTO geo_hierarchy SELECT * FROM geo_df")
conn.execute("INSERT INTO content_groups SELECT * FROM groups_df")
conn.execute("INSERT INTO hourly_active_users SELECT * FROM hourly_df")
conn.execute("INSERT INTO engagement_metrics SELECT * FROM metrics_df")
conn.execute("INSERT INTO user_engagement SELECT * FROM engagement_df")

# Create some example queries to verify the data
print("Sample hourly active users by city and content group:")
print(conn.execute("""
    SELECT 
        g.city_name,
        c.group_name,
        date_trunc('day', h.timestamp) as day,
        sum(h.active_users) as total_active_users,
        sum(h.new_users) as total_new_users
    FROM hourly_active_users h
    JOIN geo_hierarchy g ON h.city_id = g.city_id
    JOIN content_groups c ON h.group_id = c.group_id
    GROUP BY 1, 2, 3
    ORDER BY 1, 2, 3
    LIMIT 5
""").fetchdf())

print("\nEngagement metrics by content group:")
print(conn.execute("""
    SELECT 
        c.group_name,
        e.engagement_group,
        e.metric_type,
        avg(e.metric_value) as avg_value
    FROM user_engagement e
    JOIN content_groups c ON e.group_id = c.group_id
    GROUP BY 1, 2, 3
    ORDER BY 1, 2, 3
    LIMIT 5
""").fetchdf())

conn.close() 