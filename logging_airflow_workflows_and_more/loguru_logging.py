from loguru import logger
import json
import time
from datetime import utc, datetime
from typing import Dict, Any, List
import pandas as pd
import pickle


# Configure Loguru for ML production environment
def setup_ml_logging():
    """
    
    Configure Loguru for ML production logging with audit trails
    
    """
    # Remove default handler
    logger.remove()

    # Console logging for development
    logger.add(
        "stdout",
        format= "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function:{line} - {message}}",
        level="INFO"
    )

    # Production audit trail logging
    logger.add(
        "logs/ml_audit_trail_{time:YYYY-mm-DD}.log",
        rotation="1 day",
        retention="1 year",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} |{message}",
        serialize=True,
        level="INFO"
    )

    # Error logging with full stack traces 
    logger.add(
        "logs/ml_errors_{time:YYYY-MM-DD HH:mm:ss.SSS}.log",
        rotation="100 MB",
        retention="2 years",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name} : {function} : {line} | {message}",
        level="ERROR",
        backtrace=True, # Include full stack trace
        diagnose=True, # Includes variable values(good for debugging)
    )

    # Performance monitoring logs
    logger.add(
        "logs/ml_performance_monitoring_{time:YYYY-MM-DD HH:mm:ss.SSS}.log",
        rotation="500 MB",
        retention="6 months",
        filter= lambda record: "performance" in record["extra"],
        serialize=True,
        level="DEBUG"
    )

class MLModelPredictor:
    """
    
    Example ML model with comprehensive Loguru logging for audit trails
    
    """
    def __init__(self, model_path: str, model_name: str, version: str):
        self.model_path = model_path
        self.model=None        
        self.model_name = model_name
        self.version = version

        # Setup logging context
        self.logger = logger.bind(
            model_name=model_name,
            model_version=version,
            model_name=model_name,
            component="ml_predictor"
        )

        self._load_model()
    
    def _load_model(self):
        """ Load the ML model with logging """
        try:
            start_time = time.time()

            # Log the model loding start
            self.logger.info(
                "Loading model",
                extra = {
                    "action" : "model_load_start",
                    "model_path": self.model_path,
                    "timestamp": datetime.now(datetime.timezone.utc).isoformat()
                }
            )

            # Simulate model loading
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            load_time = time.time() - start_time

            # Log successful model loading with performance metrics
            self.logger.bind(performance=True).info(
                "Model loaded successfully",
                extra = {
                    "action": "model_load_success",
                    "load_time_seconds": round(load_time, 3),
                    "model_size_bytes" : 1024000,
                    "timestamp": datetime.now(datetime.timezone.utc).isoformat()
                }
            )
        except Exception as e:
            self.logger.error(
                "Failed to load model",
                extra = {
                    "action": "model_loading_faliure",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now(datetime.tiemzone.utc).isoformat()
                }
            )
            raise
    
    def predict(self, input_data: Dict[str, Any], user_id:str=None) -> Dict[str, Any]:
        """
        
        Make predictions with conprehensive audit logging
        
        This is the core function that demonstrates audit trail logging
        """
        request_id = f"req_(int(time.time() * 1000))"
        start_time = time.time()

        # Create prediction logger with context
        pred_logger = self.logger.bind(
            request_id=request_id,
            user_id=user_id or "anonymous"
        )

        try:
            # Log prediction request start (AUDIT TRAIL)
            pred_logger.info(
                "Prediction request started",
                extra = {
                    "action" : "prediction_start",
                    "input_features" : list(input_data.keys()),
                    "input_size" : len(input_data),
                    "timestamp" : datetime.now(datetime.timezone.utc).isoformat(),
                    "client_info": {
                        "request_id" : request_id,
                        "user_id" : user_id
                    }
                }
            )

            # Validate input data
            if not self._validate_input(input_data):
                pred_logger.warning(
                    "Invalid input data detected",
                    extra = {
                        "action" : "input_validation_failed",
                        "validation_errors" : "Missing required features",
                        "timestamp" : datetime.now(datetime.timezone.utc).isoformat()
                    }
                )
                raise ValueError("Invalid input data")

            prediction_result = self._make_prediction(input_data)
            prediction_time = time.time() - start_time

            # Log successful prediction (AUDIT TRAIL)
            pred_logger.info(
                "Prediction completed successfully",
                extra = {
                    "action" : "successful_prediction",
                    "prediction_result": { 
                        "prediction": prediction_result["prediction"], 
                        "confidence": prediction_result["confidence"],
                        "model_version" : self.version 
                    },
                    "performance_metrics": {
                        "prediction_time_ms" : round(prediction_time * 1000, 2),
                        "model_latency" : "acceptable"
                    },
                    "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
                    "audit_trail":{
                        "input_hash": hash(str(sorted(input_data.items()))),
                        "output_hash": hash(str(prediction_result["prediction"])),
                        "processing_node": "primary"
                    }
                }
            )

            # Log perfomance metrics seperately
            pred_logger.bind(perforance=True).debug(
                "Prediction performance metrics",
                extra = {
                    "action" : "performance_log",
                    "latency_ms" : round(prediction_time * 1000, 2),
                    "throughput" : "1 request",
                    "resource_usage" : "normal",
                    "timestamp" : datetime.now(datetime.timezone.utc).isoformat()
                }
            )

            return {
                "request_id": request_id,
                "prediction": prediction_result["prediction"],
                "confidence" : prediction_result["confidence"],
                "model_version" : self.version,
                "processing_time_ms" : round(orediction_time * 1000, 2)
            }

        except Exception as e:
            prediction_time = time.time() - start_time

            # Log prediction error
            pred_logger.error(
                "Prediction failed",
                extra = {
                    "action": "prediction_error",
                    "error_details" : {
                        "error_type" : type(e).__name__,
                        "error_message": str(e),
                        "input_data_keys" : list(input_data.keys()) if input_data else [],
                        "performance_metrics" : {
                            "failed_after_ms": round(prediction_time * 1000, 2),
                            "timestamp": datetime.now(datetime.timezone.utc),
                            "audit_trail": {
                                "faliure_point": "prediction_execution",
                                "recovery_action": "none_available"
                            }
                        }
                    } 
                }
            )
            raise

    def _validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        
        Validate input data format
        
        """
        required_features = ["feature1", "feature2", "feature3"]
        return all(feature in input_data.keys() for feature in required_features)
    
    def _make_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        
        Simulate making a prediction
        
        """
        time.sleep(0.1) # Simulate processing time
        return {
            "prediction": "positive",
            "confidence": 0.87,
            "probabilities": {"positive": 0.87, "negative": 0.13}
        }
    
    # Usage Example
    def main():
        """
        
        Example usage of ML logging system
        
        """
        # Setup logging
        setup_ml_logging()

        # Initialize model predictor
        predictor = MLModelPredictor(
            model_path="CI-CD_monitoring_testing-practice/feedback-loop/api/model.pkl",
            model_name = "LogisticRegression",
            version = "v2.1.0"
        )

        # Sample input data
        test_input = {
            "feature1": 125.50,
            "feature2" : "Credit_card",
            "feature3": 1
        }

        # Make prdiction with logging
        try:
            result = predictor.predict(
                input_data = test_input,
                user_id="user_12345"
            )
            logger.success(
                "End-to-end prediction pipeline completed",
                extra = {
                    "action": "pipeline_success",
                    "result_summary": {
                    "request_id": result["request_id"],
                    "prediction": result["prediction"],
                    "processing_time": result["processing_time_ms"],
                    "timestamp": datetime.now(datetime.timezone.utc).isoformat()
                    }
                }
            )
        except Excpetion as e:
            logger.critical(
                "Pipeline execution failed",
                extra = {
                    "action": "Pipeline faliure",
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
                    "eacalation": "required"
                }
            )
if __name__ == "__main__":
    main()
            