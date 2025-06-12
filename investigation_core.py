import networkx as nx
from typing import Dict, List, Optional, Union
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

@dataclass
class HypothesisTemplate:
    """Template for a hypothesis investigation"""
    name: str
    description: str
    query_template: str
    analyzer_class: str
    required_metrics: List[str]
    applicable_groups: Optional[List[str]] = None
    config: Optional[Dict] = None

class InvestigationGraph:
    def __init__(self):
        # Main knowledge graph
        self.graph = nx.DiGraph()
        
        # Initialize metric types that we track
        self.metric_types = {
            'daily_active_users',
            'posts_created',
            'comments_made',
            'time_spent',
            'content_shared',
            'reactions_given'
        }
        
        # Template registry
        self.hypothesis_templates: Dict[str, HypothesisTemplate] = {}
        
        # Load built-in templates
        self._load_built_in_templates()

    def _load_built_in_templates(self):
        """Load built-in hypothesis templates"""
        self.register_template(
            HypothesisTemplate(
                name="metric_anomaly",
                description="Detect significant deviations from expected metric values",
                query_template="METRIC_ANOMALY_BASE",
                analyzer_class="TimeSeriesAnalyzer",
                required_metrics=["daily_active_users"],
                config={"z_score_threshold": 3.0}
            )
        )
        
        self.register_template(
            HypothesisTemplate(
                name="segment_comparison",
                description="Compare metric values across different segments",
                query_template="SEGMENT_COMPARISON",
                analyzer_class="SegmentAnalyzer",
                required_metrics=["daily_active_users"],
                config={"alpha": 0.05, "min_segment_size": 30}
            )
        )
        
        self.register_template(
            HypothesisTemplate(
                name="metric_correlation",
                description="Analyze correlations between different metrics",
                query_template="CORRELATION_ANALYSIS",
                analyzer_class="CorrelationAnalyzer",
                required_metrics=["daily_active_users", "posts_created"],
                config={"correlation_threshold": 0.7}
            )
        )

    def register_template(self, template: HypothesisTemplate) -> None:
        """Register a new hypothesis template"""
        self.hypothesis_templates[template.name] = template

    def add_hypothesis(self, 
                      template_name: str,
                      params: Dict[str, Any] = None) -> str:
        """
        Add a new hypothesis node based on a template.
        
        Args:
            template_name: Name of the template to use
            params: Parameters to customize the template
            
        Returns:
            hypothesis_id: Unique identifier for the hypothesis
        """
        if template_name not in self.hypothesis_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.hypothesis_templates[template_name]
        params = params or {}
        
        # Generate unique ID for the hypothesis
        hypothesis_id = hashlib.sha256(
            f"{template_name}{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()[:12]
        
        # Create hypothesis node
        self.graph.add_node(hypothesis_id, 
                           type='hypothesis',
                           template_name=template_name,
                           params=params,
                           created_at=datetime.now().isoformat(),
                           success_count=0,
                           failure_count=0,
                           last_used=None)
        
        # Connect to relevant metrics
        for metric in template.required_metrics:
            if metric in self.metric_types:
                self.graph.add_edge(hypothesis_id, f"metric_{metric}", 
                                  relationship_type='analyzes')
        
        # Connect to content groups if specified
        if template.applicable_groups:
            for group in template.applicable_groups:
                self.graph.add_edge(hypothesis_id, f"group_{group}",
                                  relationship_type='applies_to')
        
        return hypothesis_id

    def update_hypothesis_result(self, 
                               hypothesis_id: str, 
                               success: bool,
                               findings: Dict = None) -> None:
        """
        Update hypothesis node with execution results.
        
        Args:
            hypothesis_id: ID of the hypothesis
            success: Whether the hypothesis helped explain the anomaly
            findings: Optional dict with detailed findings
        """
        if hypothesis_id in self.graph:
            node = self.graph.nodes[hypothesis_id]
            if success:
                node['success_count'] = node.get('success_count', 0) + 1
            else:
                node['failure_count'] = node.get('failure_count', 0) + 1
            
            node['last_used'] = datetime.now().isoformat()
            if findings:
                node['latest_findings'] = findings

    def get_relevant_hypotheses(self, 
                              metrics: List[str],
                              groups: Optional[List[str]] = None,
                              limit: int = 5) -> List[Dict]:
        """
        Get relevant hypotheses for investigating given metrics and groups.
        
        Args:
            metrics: List of metrics showing anomalies
            groups: Optional list of content groups involved
            limit: Maximum number of hypotheses to return
            
        Returns:
            List of relevant hypothesis nodes, sorted by success rate
        """
        relevant_hypotheses = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'hypothesis':
                # Check if hypothesis template handles these metrics
                template = self.hypothesis_templates.get(
                    self.graph.nodes[node].get('template_name')
                )
                if not template:
                    continue
                
                metric_match = any(
                    metric in template.required_metrics
                    for metric in metrics
                )
                
                # Check if hypothesis is applicable to these groups
                group_match = True
                if groups and template.applicable_groups:
                    group_match = any(
                        group in template.applicable_groups
                        for group in groups
                    )
                
                if metric_match and group_match:
                    node_data = self.graph.nodes[node].copy()
                    # Calculate success rate
                    total = node_data.get('success_count', 0) + node_data.get('failure_count', 0)
                    success_rate = node_data.get('success_count', 0) / total if total > 0 else 0
                    node_data['success_rate'] = success_rate
                    node_data['id'] = node
                    relevant_hypotheses.append(node_data)
        
        # Sort by success rate and limit results
        return sorted(relevant_hypotheses, 
                     key=lambda x: x['success_rate'],
                     reverse=True)[:limit]

    def export_graph(self, filepath: str) -> None:
        """Export the knowledge graph to a JSON file."""
        graph_data = nx.node_link_data(self.graph)
        # Add templates to the export
        graph_data['templates'] = {
            name: vars(template)
            for name, template in self.hypothesis_templates.items()
        }
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)

    def import_graph(self, filepath: str) -> None:
        """Import the knowledge graph from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load templates first
        templates = data.pop('templates', {})
        for name, template_data in templates.items():
            self.register_template(HypothesisTemplate(**template_data))
        
        # Then load the graph
        self.graph = nx.node_link_graph(data)

    def get_hypothesis_details(self, hypothesis_id: str) -> Optional[Dict]:
        """Get full details of a specific hypothesis."""
        if hypothesis_id in self.graph:
            node_data = self.graph.nodes[hypothesis_id]
            template = self.hypothesis_templates.get(node_data.get('template_name'))
            if template:
                return {
                    'id': hypothesis_id,
                    'template': template,
                    **node_data
                }
        return None

    def get_graph_stats(self) -> Dict:
        """Get basic statistics about the knowledge graph."""
        hypothesis_nodes = [n for n, d in self.graph.nodes(data=True) 
                          if d.get('type') == 'hypothesis']
        
        total_executions = sum(
            self.graph.nodes[n].get('success_count', 0) + 
            self.graph.nodes[n].get('failure_count', 0)
            for n in hypothesis_nodes
        )
        
        return {
            'total_hypotheses': len(hypothesis_nodes),
            'total_executions': total_executions,
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges()
        } 