from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import asyncio
import traceback
from datetime import datetime
import logging
from enum import Enum

class ExecutionStatus(Enum):
    SUCCESS = "success"
    TECHNICAL_FAILURE = "technical_failure"
    NO_FINDINGS = "no_findings"
    INVALID_HYPOTHESIS = "invalid_hypothesis"

class TechnicalErrorType(Enum):
    SQL_ERROR = "sql_error"
    PYTHON_ERROR = "python_error"
    DATA_ACCESS_ERROR = "data_access_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"

@dataclass
class TechnicalError:
    """Represents a technical execution error"""
    error_type: TechnicalErrorType
    error_message: str
    stack_trace: str
    timestamp: str
    query_id: Optional[str] = None
    script_id: Optional[str] = None

@dataclass
class ValidationResult:
    """Represents the validity check of a hypothesis"""
    is_valid: bool
    confidence_score: float
    findings: Dict
    validation_metrics: Dict
    explanation: str

@dataclass
class ExecutionResult:
    """Complete result of hypothesis execution"""
    hypothesis_id: str
    investigation_id: str
    status: ExecutionStatus
    execution_time: float
    timestamp: str
    technical_error: Optional[TechnicalError] = None
    validation_result: Optional[ValidationResult] = None
    alerts_sent: List[str] = None

class ExecutionHandler:
    def __init__(self, alert_config: Dict = None):
        self.alert_config = alert_config or {
            'alert_threshold': 3,  # Number of failures before marking as broken
            'timeout_seconds': 300,  # 5 minutes
            'max_memory_mb': 1024,
        }
        self.error_counts: Dict[str, Dict] = {}  # Track error counts per hypothesis
        self.logger = logging.getLogger(__name__)

    async def execute_hypothesis(self,
                               hypothesis_id: str,
                               investigation_id: str,
                               query: str,
                               analysis_script: str,
                               params: Dict) -> ExecutionResult:
        """
        Execute a hypothesis with two-tier error handling.
        
        Args:
            hypothesis_id: ID of the hypothesis
            investigation_id: ID of the investigation
            query: SQL query to execute
            analysis_script: Python analysis script
            params: Execution parameters
        
        Returns:
            ExecutionResult containing status and details
        """
        start_time = datetime.now()
        
        try:
            # First tier: Technical execution
            query_result = await self._execute_query(query, params)
            analysis_result = await self._execute_analysis(analysis_script, query_result)
            
            # Second tier: Hypothesis validation
            validation = self._validate_hypothesis(analysis_result, params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Reset error count on successful execution
            if hypothesis_id in self.error_counts:
                self.error_counts[hypothesis_id] = {'count': 0, 'errors': []}
            
            return ExecutionResult(
                hypothesis_id=hypothesis_id,
                investigation_id=investigation_id,
                status=ExecutionStatus.SUCCESS if validation.is_valid else ExecutionStatus.NO_FINDINGS,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                validation_result=validation
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error = self._handle_technical_error(e, hypothesis_id)
            
            return ExecutionResult(
                hypothesis_id=hypothesis_id,
                investigation_id=investigation_id,
                status=ExecutionStatus.TECHNICAL_FAILURE,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat(),
                technical_error=error,
                alerts_sent=self._send_alerts(error, hypothesis_id)
            )

    async def _execute_query(self, query: str, params: Dict) -> Dict:
        """Execute SQL query with error handling"""
        try:
            # Your SQL execution logic here
            # This is a placeholder
            return {'data': []}
        except Exception as e:
            error_type = TechnicalErrorType.SQL_ERROR
            if 'timeout' in str(e).lower():
                error_type = TechnicalErrorType.TIMEOUT_ERROR
            elif 'memory' in str(e).lower():
                error_type = TechnicalErrorType.MEMORY_ERROR
            raise Exception(f"{error_type.value}: {str(e)}")

    async def _execute_analysis(self, script: str, data: Dict) -> Dict:
        """Execute Python analysis with error handling"""
        try:
            # Your Python execution logic here
            # This is a placeholder
            return {'results': []}
        except Exception as e:
            raise Exception(f"{TechnicalErrorType.PYTHON_ERROR.value}: {str(e)}")

    def _validate_hypothesis(self, 
                           analysis_result: Dict, 
                           params: Dict) -> ValidationResult:
        """
        Validate hypothesis results against expected outcomes.
        This is where we check if the hypothesis actually explains anything,
        separate from technical execution success.
        """
        # Example validation logic
        findings = analysis_result.get('results', [])
        confidence_threshold = params.get('confidence_threshold', 0.5)
        
        if not findings:
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                findings={},
                validation_metrics={'finding_count': 0},
                explanation="No findings generated"
            )
        
        # Calculate validation metrics
        metrics = {
            'finding_count': len(findings),
            'average_confidence': sum(f.get('confidence', 0) for f in findings) / len(findings),
            'max_correlation': max(f.get('correlation', 0) for f in findings),
        }
        
        is_valid = metrics['average_confidence'] >= confidence_threshold
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=metrics['average_confidence'],
            findings=findings,
            validation_metrics=metrics,
            explanation=self._generate_validation_explanation(metrics, is_valid)
        )

    def _handle_technical_error(self, 
                              error: Exception, 
                              hypothesis_id: str) -> TechnicalError:
        """Handle and categorize technical errors"""
        error_msg = str(error)
        error_type = TechnicalErrorType.PYTHON_ERROR  # Default
        
        # Categorize error
        if "SQL" in error_msg or "query" in error_msg.lower():
            error_type = TechnicalErrorType.SQL_ERROR
        elif "timeout" in error_msg.lower():
            error_type = TechnicalErrorType.TIMEOUT_ERROR
        elif "memory" in error_msg.lower():
            error_type = TechnicalErrorType.MEMORY_ERROR
        
        # Track error count for this hypothesis
        if hypothesis_id not in self.error_counts:
            self.error_counts[hypothesis_id] = {'count': 0, 'errors': []}
        
        self.error_counts[hypothesis_id]['count'] += 1
        self.error_counts[hypothesis_id]['errors'].append({
            'type': error_type.value,
            'message': error_msg,
            'timestamp': datetime.now().isoformat()
        })
        
        return TechnicalError(
            error_type=error_type,
            error_message=error_msg,
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now().isoformat()
        )

    def _send_alerts(self, 
                    error: TechnicalError, 
                    hypothesis_id: str) -> List[str]:
        """Send alerts for technical failures"""
        alerts_sent = []
        error_count = self.error_counts[hypothesis_id]['count']
        
        # Alert on first occurrence of critical errors
        if error.error_type in [TechnicalErrorType.MEMORY_ERROR, 
                              TechnicalErrorType.TIMEOUT_ERROR]:
            alerts_sent.append(self._send_immediate_alert(error, hypothesis_id))
        
        # Alert when error count exceeds threshold
        if error_count >= self.alert_config['alert_threshold']:
            alerts_sent.append(
                self._send_broken_hypothesis_alert(hypothesis_id, 
                                                 self.error_counts[hypothesis_id])
            )
        
        return alerts_sent

    def _send_immediate_alert(self, 
                            error: TechnicalError, 
                            hypothesis_id: str) -> str:
        """Send immediate alert for critical errors"""
        alert_message = (
            f"CRITICAL: Hypothesis {hypothesis_id} failed with {error.error_type.value}\n"
            f"Error: {error.error_message}\n"
            f"Time: {error.timestamp}"
        )
        
        self.logger.error(alert_message)
        # Your alert sending logic here (e.g., email, Slack, etc.)
        return f"immediate_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _send_broken_hypothesis_alert(self, 
                                    hypothesis_id: str, 
                                    error_history: Dict) -> str:
        """Send alert when hypothesis is marked as broken"""
        alert_message = (
            f"WARNING: Hypothesis {hypothesis_id} marked as broken\n"
            f"Failed {error_history['count']} times\n"
            f"Recent errors: {error_history['errors'][-3:]}"  # Last 3 errors
        )
        
        self.logger.warning(alert_message)
        # Your alert sending logic here
        return f"broken_hypothesis_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _generate_validation_explanation(self, 
                                      metrics: Dict, 
                                      is_valid: bool) -> str:
        """Generate human-readable explanation of validation results"""
        if not is_valid:
            return (
                f"Hypothesis generated {metrics['finding_count']} findings but "
                f"confidence ({metrics['average_confidence']:.2f}) is too low. "
                f"Max correlation observed: {metrics['max_correlation']:.2f}"
            )
        
        return (
            f"Hypothesis successfully explained the anomaly with "
            f"{metrics['finding_count']} findings. "
            f"Average confidence: {metrics['average_confidence']:.2f}, "
            f"Max correlation: {metrics['max_correlation']:.2f}"
        ) 