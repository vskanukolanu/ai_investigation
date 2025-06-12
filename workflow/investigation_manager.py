from typing import List, Dict, Optional, Tuple
import asyncio
from datetime import datetime
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
from .execution_handler import (
    ExecutionHandler, ExecutionStatus, ExecutionResult, 
    ValidationResult, TechnicalError
)

@dataclass
class MetricAnomaly:
    """Represents a detected metric anomaly"""
    metric_name: str
    change_percentage: float
    time_period: str
    groups: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

@dataclass
class HypothesisScore:
    """Stores various components of hypothesis scoring"""
    base_relevance: float
    selection_score: float
    execution_score: float
    context_score: float
    temporal_score: float
    final_score: float

class InvestigationManager:
    def __init__(self, graph, learning_config: Dict = None):
        self.graph = graph
        self.config = learning_config or {
            'selection_weight': 0.3,
            'execution_weight': 0.2,
            'context_weight': 0.2,
            'temporal_weight': 0.15,
            'relationship_weight': 0.15,
            'temporal_decay_days': 30,
            'min_confidence': 0.1
        }
        
        # Initialize execution handler
        self.execution_handler = ExecutionHandler()
        
        # Learning storage
        self.selection_history: List[Dict] = []
        self.execution_history: List[Dict] = []
        self.context_history: Dict[str, List[Dict]] = {}
        self.relationship_scores: Dict[Tuple[str, str], float] = {}
        
        # Track broken hypotheses
        self.broken_hypotheses: Dict[str, Dict] = {}
        
        # Load historical data if exists
        self._load_learning_data()

    def initiate_investigation(self, anomaly: MetricAnomaly) -> Dict:
        """
        Start investigation workflow for an anomaly.
        
        Args:
            anomaly: Details of the metric anomaly
            
        Returns:
            Dict containing ranked hypotheses and metadata
        """
        # Get relevant hypotheses
        hypotheses = self.graph.get_relevant_hypotheses(
            metrics=[anomaly.metric_name],
            groups=anomaly.groups
        )
        
        # Score and rank hypotheses
        ranked_hypotheses = self._rank_hypotheses(hypotheses, anomaly)
        
        return {
            'anomaly': anomaly,
            'hypotheses': ranked_hypotheses,
            'timestamp': datetime.now().isoformat(),
            'investigation_id': self._generate_investigation_id(anomaly)
        }

    def _rank_hypotheses(self, 
                        hypotheses: List[Dict], 
                        anomaly: MetricAnomaly) -> List[Dict]:
        """Rank hypotheses using multiple learning mechanisms"""
        scored_hypotheses = []
        
        for hypothesis in hypotheses:
            # Calculate various scores
            selection_score = self._get_selection_score(hypothesis['id'])
            execution_score = self._get_execution_score(hypothesis['id'])
            context_score = self._get_context_score(hypothesis['id'], anomaly)
            temporal_score = self._get_temporal_score(hypothesis['id'])
            relationship_score = self._get_relationship_score(hypothesis['id'], hypotheses)
            
            # Combine scores using weights
            final_score = (
                selection_score * self.config['selection_weight'] +
                execution_score * self.config['execution_weight'] +
                context_score * self.config['context_weight'] +
                temporal_score * self.config['temporal_weight'] +
                relationship_score * self.config['relationship_weight']
            )
            
            scored_hypotheses.append({
                **hypothesis,
                'scores': HypothesisScore(
                    base_relevance=hypothesis.get('success_rate', 0),
                    selection_score=selection_score,
                    execution_score=execution_score,
                    context_score=context_score,
                    temporal_score=temporal_score,
                    final_score=final_score
                )
            })
        
        # Sort by final score
        return sorted(scored_hypotheses, 
                     key=lambda x: x['scores'].final_score, 
                     reverse=True)

    async def execute_hypotheses(self, 
                               investigation_id: str,
                               hypothesis_ids: List[str]) -> Dict:
        """Execute selected hypotheses with two-tier error handling"""
        tasks = []
        results = []
        
        for h_id in hypothesis_ids:
            # Skip broken hypotheses
            if h_id in self.broken_hypotheses:
                results.append({
                    'hypothesis_id': h_id,
                    'status': 'SKIPPED',
                    'reason': 'Hypothesis marked as broken',
                    'error_history': self.broken_hypotheses[h_id]
                })
                continue
            
            hypothesis = self.graph.get_hypothesis_details(h_id)
            if not hypothesis:
                continue
            
            tasks.append(self._execute_single_hypothesis(investigation_id, hypothesis))
        
        execution_results = await asyncio.gather(*tasks)
        results.extend(execution_results)
        
        # Update learning based on results
        self._update_learning_from_results(results)
        
        return self._generate_report(results)

    async def _execute_single_hypothesis(self, 
                                      investigation_id: str, 
                                      hypothesis: Dict) -> ExecutionResult:
        """Execute a single hypothesis with error handling"""
        try:
            result = await self.execution_handler.execute_hypothesis(
                hypothesis_id=hypothesis['id'],
                investigation_id=investigation_id,
                query=hypothesis['template'].query_template,
                analysis_script=hypothesis['template'].analyzer_class,
                params=hypothesis['params']
            )
            
            # Handle technical failures
            if result.status == ExecutionStatus.TECHNICAL_FAILURE:
                self._handle_technical_failure(hypothesis['id'], result)
            
            # Track execution result
            self.execution_history.append({
                'hypothesis_id': hypothesis['id'],
                'investigation_id': investigation_id,
                'timestamp': result.timestamp,
                'status': result.status.value,
                'execution_time': result.execution_time,
                'validation_result': result.validation_result.__dict__ if result.validation_result else None
            })
            
            return result
            
        except Exception as e:
            # Catch any unexpected errors
            return ExecutionResult(
                hypothesis_id=hypothesis['id'],
                investigation_id=investigation_id,
                status=ExecutionStatus.TECHNICAL_FAILURE,
                execution_time=0,
                timestamp=datetime.now().isoformat(),
                technical_error=TechnicalError(
                    error_type="UNEXPECTED_ERROR",
                    error_message=str(e),
                    stack_trace="",
                    timestamp=datetime.now().isoformat()
                )
            )

    def _handle_technical_failure(self, 
                                hypothesis_id: str, 
                                result: ExecutionResult) -> None:
        """Handle technical failures and mark hypotheses as broken if needed"""
        if hypothesis_id not in self.broken_hypotheses:
            self.broken_hypotheses[hypothesis_id] = {
                'first_error': result.timestamp,
                'error_count': 0,
                'recent_errors': []
            }
        
        broken_info = self.broken_hypotheses[hypothesis_id]
        broken_info['error_count'] += 1
        broken_info['recent_errors'].append({
            'timestamp': result.timestamp,
            'error_type': result.technical_error.error_type,
            'error_message': result.technical_error.error_message
        })
        
        # Keep only last 5 errors
        broken_info['recent_errors'] = broken_info['recent_errors'][-5:]
        
        # Save broken hypotheses state
        self._save_broken_hypotheses()

    def _update_learning_from_results(self, results: List[ExecutionResult]) -> None:
        """Update learning system based on execution results"""
        for result in results:
            if result.status == ExecutionStatus.SUCCESS:
                # Update context learning
                if result.validation_result and result.validation_result.is_valid:
                    self._update_context_history(
                        result.hypothesis_id,
                        result.validation_result
                    )
                
                # Update relationship learning
                self._update_relationship_scores(results)
            
            elif result.status == ExecutionStatus.TECHNICAL_FAILURE:
                # Technical failures are handled separately
                continue
            
            elif result.status == ExecutionStatus.NO_FINDINGS:
                # Update confidence scores for invalid hypotheses
                self._update_confidence_scores(
                    result.hypothesis_id,
                    result.validation_result
                )
        
        self._save_learning_data()

    def _update_confidence_scores(self, 
                                hypothesis_id: str, 
                                validation: ValidationResult) -> None:
        """Update confidence scores for hypothesis based on validation results"""
        if not validation:
            return
        
        # Find historical executions for this hypothesis
        historical = [
            e for e in self.execution_history 
            if e['hypothesis_id'] == hypothesis_id
        ]
        
        # Calculate new confidence score
        current_confidence = validation.confidence_score
        historical_confidence = np.mean([
            e['validation_result']['confidence_score'] 
            for e in historical 
            if e['validation_result']
        ]) if historical else current_confidence
        
        # Update hypothesis confidence in graph
        self.graph.update_hypothesis_confidence(
            hypothesis_id,
            (historical_confidence + current_confidence) / 2
        )

    def _update_context_history(self, 
                              hypothesis_id: str, 
                              validation: ValidationResult) -> None:
        """Update context history with successful executions"""
        if hypothesis_id not in self.context_history:
            self.context_history[hypothesis_id] = []
        
        self.context_history[hypothesis_id].append({
            'timestamp': datetime.now().isoformat(),
            'confidence_score': validation.confidence_score,
            'validation_metrics': validation.validation_metrics,
            'keywords': self._extract_keywords(validation.findings)
        })

    def _update_relationship_scores(self, results: List[ExecutionResult]) -> None:
        """Update relationship scores between hypotheses"""
        successful_hypotheses = [
            r.hypothesis_id for r in results 
            if r.status == ExecutionStatus.SUCCESS and 
            r.validation_result and 
            r.validation_result.is_valid
        ]
        
        # Update scores for all pairs of successful hypotheses
        for i, h1 in enumerate(successful_hypotheses):
            for h2 in successful_hypotheses[i+1:]:
                pair_key = tuple(sorted([h1, h2]))
                
                # Increase relationship score
                current_score = self.relationship_scores.get(pair_key, 0.0)
                self.relationship_scores[pair_key] = min(1.0, current_score + 0.1)

    def _save_broken_hypotheses(self) -> None:
        """Save broken hypotheses state"""
        data_path = Path("learning_data")
        data_path.mkdir(exist_ok=True)
        
        with open(data_path / "broken_hypotheses.json", 'w') as f:
            json.dump(self.broken_hypotheses, f)

    def _load_broken_hypotheses(self) -> None:
        """Load broken hypotheses state"""
        data_path = Path("learning_data/broken_hypotheses.json")
        if data_path.exists():
            with open(data_path, 'r') as f:
                self.broken_hypotheses = json.load(f)

    def _generate_report(self, results: List[ExecutionResult]) -> Dict:
        """Generate detailed execution report"""
        successful_results = [
            r for r in results 
            if r.status == ExecutionStatus.SUCCESS
        ]
        technical_failures = [
            r for r in results 
            if r.status == ExecutionStatus.TECHNICAL_FAILURE
        ]
        invalid_results = [
            r for r in results 
            if r.status == ExecutionStatus.NO_FINDINGS
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_hypotheses': len(results),
            'successful_executions': len(successful_results),
            'technical_failures': {
                'count': len(technical_failures),
                'details': [
                    {
                        'hypothesis_id': r.hypothesis_id,
                        'error_type': r.technical_error.error_type,
                        'error_message': r.technical_error.error_message
                    }
                    for r in technical_failures
                ]
            },
            'invalid_hypotheses': {
                'count': len(invalid_results),
                'details': [
                    {
                        'hypothesis_id': r.hypothesis_id,
                        'confidence': r.validation_result.confidence_score if r.validation_result else 0,
                        'explanation': r.validation_result.explanation if r.validation_result else None
                    }
                    for r in invalid_results
                ]
            },
            'findings': [
                {
                    'hypothesis_id': r.hypothesis_id,
                    'confidence': r.validation_result.confidence_score,
                    'findings': r.validation_result.findings,
                    'metrics': r.validation_result.validation_metrics
                }
                for r in successful_results
                if r.validation_result
            ]
        }

    def _generate_investigation_id(self, anomaly: MetricAnomaly) -> str:
        """Generate unique investigation ID"""
        return f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{anomaly.metric_name}"

    def _load_learning_data(self) -> None:
        """Load historical learning data"""
        data_path = Path("learning_data")
        if data_path.exists():
            with open(data_path / "selection_history.json", 'r') as f:
                self.selection_history = json.load(f)
            with open(data_path / "execution_history.json", 'r') as f:
                self.execution_history = json.load(f)
            with open(data_path / "context_history.json", 'r') as f:
                self.context_history = json.load(f)
            with open(data_path / "relationship_scores.json", 'r') as f:
                self.relationship_scores = json.load(f)

    def _save_learning_data(self) -> None:
        """Save learning data to disk"""
        data_path = Path("learning_data")
        data_path.mkdir(exist_ok=True)
        
        with open(data_path / "selection_history.json", 'w') as f:
            json.dump(self.selection_history, f)
        with open(data_path / "execution_history.json", 'w') as f:
            json.dump(self.execution_history, f)
        with open(data_path / "context_history.json", 'w') as f:
            json.dump(self.context_history, f)
        with open(data_path / "relationship_scores.json", 'w') as f:
            json.dump(self.relationship_scores, f) 