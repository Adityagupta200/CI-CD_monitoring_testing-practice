import logging
import json
from datetime import datetime
from src.monitoring import get_model_metrics
from src.deployment import deploy_model, get_previous_model_version

class AutomatedRollback:
    def __init__(self, performance_threshold=0.85, monitoring_window=300):
        self.performance_threshold = performance_threshold
        self.monitoring_window = monitoring_window
        self.logger = logging.getLogger(__name__)

    def monitor_and_rollback(self):
        """Monitor current model and trigger rollback if needed"""
        try:
            # Get current model performance metrics
            current_metrics = get_model_metrics(window_seconds=self.monitoring_window)

            # Check if performance has degraded
            if current_metrics['accuracy'] < self.performance_threshold:
                self.logger.warning(
                    f"Model performance degraded: {current_metrics['accuracy']:.3f}" f"< {self.performance_threshold}")
            
                # Trigger automated rollback
                self.execute_rollback()
                return True
            return False
        
        except Exception as e:
            self.logger.error(f"Error during monitoring: {e}")
            # In case of monitoring failure, trigger rollback for safety
            self.execute_rollback()
            return False
    def execute_rollback(self):
        """Execture rollback to previous stable model version"""
        try:
            # Get previous stable model version
            previous_version = get_previous_model_version()

            self.logger.info(f"Rolling back to model version: {previous_version}")

            # Deploy previous model version
            deploy_model(version=previous_version)

            # Log rollback event
            rollback_log = {
                'timestamp': datetime.now().isoformat(),
                'action': 'automated_rollback',
                'previous_version': previous_version,
                'reason': 'preformance_degradation'
            }

            # Save rollback log
            with open('rollback_log.json', 'w') as f:
                f.write(json.dumps(rollback_log) + '\n')
            self.logger.info("Automated rollback completed successfully")
        
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            raise

# Usage in monitoring script
if __name__ == "__main__":
    rollback_system = AutomatedRollback(performance_threshold=0.90)
    rollback_triggered = rollback_system.monitor_an_rollback()

    if rollback_triggered:
        print("Automated Rollback Executed")
    else:
        print("No rollback triggered")