from typing import Dict, Any
from jinja2 import Template

class QueryTemplate:
    """Base class for SQL query templates"""
    
    def __init__(self, template_str: str):
        self.template = Template(template_str)
    
    def render(self, params: Dict[str, Any]) -> str:
        """Render the query with given parameters"""
        return self.template.render(**params)

# Common query templates
METRIC_ANOMALY_BASE = QueryTemplate("""
    WITH metric_stats AS (
        SELECT 
            timestamp,
            {{ metric_name }},
            AVG({{ metric_name }}) OVER (
                ORDER BY timestamp
                ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
            ) as expected_value,
            STDDEV({{ metric_name }}) OVER (
                ORDER BY timestamp
                ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING
            ) as std_dev
        FROM {{ table_name }}
        WHERE timestamp BETWEEN '{{ start_date }}' AND '{{ end_date }}'
        {% if group_id %}
        AND group_id = {{ group_id }}
        {% endif %}
        {% if city_id %}
        AND city_id = {{ city_id }}
        {% endif %}
    )
    SELECT 
        timestamp,
        {{ metric_name }},
        expected_value,
        ({{ metric_name }} - expected_value) / NULLIF(std_dev, 0) as z_score
    FROM metric_stats
    WHERE ABS(({{ metric_name }} - expected_value) / NULLIF(std_dev, 0)) > {{ threshold }}
    ORDER BY timestamp
""")

SEGMENT_COMPARISON = QueryTemplate("""
    WITH segment_metrics AS (
        SELECT 
            timestamp,
            {{ segment_column }},
            SUM({{ metric_name }}) as metric_value
        FROM {{ table_name }}
        WHERE timestamp BETWEEN '{{ start_date }}' AND '{{ end_date }}'
        GROUP BY timestamp, {{ segment_column }}
    )
    SELECT 
        {{ segment_column }},
        AVG(metric_value) as avg_value,
        STDDEV(metric_value) as std_dev,
        COUNT(*) as sample_size
    FROM segment_metrics
    GROUP BY {{ segment_column }}
    ORDER BY avg_value DESC
""")

CORRELATION_ANALYSIS = QueryTemplate("""
    SELECT 
        CORR(m1.{{ metric1 }}, m2.{{ metric2 }}) as correlation,
        COUNT(*) as sample_size
    FROM {{ table1 }} m1
    JOIN {{ table2 }} m2 
    ON m1.timestamp = m2.timestamp
    WHERE m1.timestamp BETWEEN '{{ start_date }}' AND '{{ end_date }}'
    {% if group_id %}
    AND m1.group_id = {{ group_id }}
    {% endif %}
""") 