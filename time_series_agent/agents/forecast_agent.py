"""
Forecast Agent for Time Series Prediction
Predict Agent - responsible for predicting, ensemble prediction, and prediction visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import json

from agents.memory import ExperimentMemory
from utils.visualization_utils import TimeSeriesVisualizer
from utils.model_library import MODEL_FUNCTIONS, get_model_function
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORECAST_SYSTEM_PROMPT = """
You are the Ensemble Forecasting Integration Agent for a high-stakes time series prediction system.

Background:
- You are an expert in ensemble methods, model averaging, and uncertainty quantification for time series forecasting.
- Your integration strategy can significantly impact the accuracy and reliability of the final forecast.

Your responsibilities:
- Review the individual model forecasts and any available visualizations.
- Decide the most appropriate ensemble integration strategy (e.g., best model, weighted average, trimmed mean, median, custom weights).
- If using weights, specify them and explain your rationale.
- Justify your integration choice, considering model diversity, agreement, and historical performance.
- Assess your confidence in the ensemble and note any risks or caveats.
- Always return your decision in a structured Python dict, with transparent reasoning.

You have access to:
- The individual model forecasts (as a Python dict)
- Visualizations of the forecasts and historical data
- Prediction tools for different models (ARMA, LSTM, RandomForest, etc.)

Your output will be used as the final forecast for this time series slice.
"""

def get_ensemble_decision_prompt(individual_forecasts: dict, visualizations: dict = None) -> str:
    import json
    viz_info = ""
    if visualizations:
        viz_info = f"\nVisualizations:\n{visualizations}\n"
    return f"""
You are an ensemble forecasting expert.

Given the following individual model forecasts:
{json.dumps(individual_forecasts, indent=2)}
{viz_info}

Please:
1. Decide the best ensemble integration strategy (choose from: best_model, weighted_average, trimmed_mean, median, custom_weights).
2. If using weights, specify the weights for each model.
3. Justify your choice.
4. Assess your confidence in the ensemble.

IMPORTANT: Return your answer ONLY as a JSON object, with NO markdown formatting, NO code blocks, NO explanations. Just the raw JSON:
{{
  "integration_strategy": "string",
  "weights": {{"model_name": "float"}} (if applicable),
  "selected_model": "string" (if best_model),
  "reasoning": "string",
  "confidence": "string"
}}
"""

class ForecastAgent:
    """
    Predict Agent
    Responsible for predicting, ensemble prediction, and prediction visualization
    """
    
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            # max_tokens=4000,
        )
        self.config = config or {}
        self.visualizer = TimeSeriesVisualizer(self.config)
        self.memory = ExperimentMemory(self.config)
        
        # Store model functions from library
        self.model_functions = MODEL_FUNCTIONS

    def run(self, selected_models: list, best_hyperparameters: dict, validation_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str = None, validation_metrics: dict = None):
        """
        Run the forecast agent to generate predictions using validation data and calculate metrics against test data
        
        Args:
            selected_models: List of selected model names
            best_hyperparameters: Dictionary of best hyperparameters for each model
            validation_data: Validation dataset for training models
            test_data: Test dataset for final evaluation
            output_dir: Output directory for saving results
            validation_metrics: Dictionary containing validation metrics (MAE, MSE, MAPE) for each model
            
        Returns:
            Dictionary containing individual predictions, ensemble predictions, and metrics
        """
        import time
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        logger.info(f"Starting forecast using validation data with {len(selected_models)} models")
        
        # Generate individual model predictions using validation data
        individual_predictions = {}
        for model_name in selected_models:
            try:
                logger.info(f"Generating predictions for {model_name}")
                hyperparams = best_hyperparameters.get(model_name, {})
                
                # Get model function and generate predictions
                model_func = get_model_function(model_name)
                data_dict = {'value': validation_data['value'].values}
                horizon = len(test_data)
                
                predictions = model_func(data_dict, hyperparams, horizon)
                individual_predictions[model_name] = predictions
                
                logger.info(f"Generated {len(predictions)} predictions for {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to generate predictions for {model_name}: {e}")
                # Generate fallback predictions
                fallback_predictions = self._generate_fallback_predictions(test_data)
                individual_predictions[model_name] = fallback_predictions
        
        # Extract validation MAE from validation_metrics
        validation_mae = {}
        if validation_metrics:
            for model_name in selected_models:
                if model_name in validation_metrics:
                    if isinstance(validation_metrics[model_name], dict) and 'mae' in validation_metrics[model_name]:
                        validation_mae[model_name] = validation_metrics[model_name]['mae']
                    else:
                        # If validation_metrics[model_name] is directly the MAE value
                        validation_mae[model_name] = validation_metrics[model_name]
        
        logger.info(f"Validation MAE data: {validation_mae}")
        
        # Generate ensemble predictions
        ensemble_predictions = self._generate_ensemble_predictions(individual_predictions, validation_mae)
        
        # Calculate forecast metrics
        forecast_metrics = self._calculate_forecast_metrics(individual_predictions, ensemble_predictions)
        
        # Calculate test set metrics (MSE, MAE, MAPE)
        test_metrics = self._calculate_test_metrics(individual_predictions, ensemble_predictions, test_data)
        
        # Generate confidence intervals
        confidence_intervals = self._generate_confidence_intervals(individual_predictions, ensemble_predictions)
        
        # Generate visualizations
        visualizations = self._generate_forecast_visualizations(
            validation_data, individual_predictions, ensemble_predictions, 
            confidence_intervals, test_data, output_dir
        )
        
        # Save results
        self._save_forecast_results(
            individual_predictions, ensemble_predictions, forecast_metrics,
            confidence_intervals, output_dir
        )
        
        # Update memory
        self._update_memory(
            individual_predictions, ensemble_predictions, forecast_metrics,
            confidence_intervals, visualizations
        )
        
        result = {
            'individual_predictions': individual_predictions,
            'ensemble_predictions': ensemble_predictions,
            'forecast_metrics': forecast_metrics,
            'confidence_intervals': confidence_intervals,
            'test_metrics': test_metrics,
            'visualizations': visualizations
        }
        
        logger.info("Forecast completed successfully")
        return result
    
    def _generate_fallback_predictions(self, test_data: pd.DataFrame) -> List[float]:
        """Generate fallback predictions when model fails"""
        n_predictions = len(test_data)
        # Simple moving average based prediction
        mean_value = test_data['value'].mean()
        std_value = test_data['value'].std()
        
        predictions = []
        for i in range(n_predictions):
            # Add some randomness to avoid identical predictions
            pred = mean_value + np.random.normal(0, std_value * 0.1)
            predictions.append(max(0, pred))  # Ensure non-negative
        
        return predictions
    
    def _generate_ensemble_predictions(self, individual_predictions: Dict[str, List[float]], validation_mae: Dict[str, float] = None) -> Dict[str, Any]:
        """Generate ensemble predictions"""
        logger.info("Generating ensemble predictions...")
        
        if not individual_predictions:
            return {}
        
        # Calculate ensemble predictions
        predictions_array = np.array(list(individual_predictions.values()))
        
        ensemble_results = {}
        
        # Simple average
        ensemble_results['simple_average'] = np.mean(predictions_array, axis=0).tolist()
        
        # Weighted average (based on model performance using LLM)
        weights = self._calculate_model_weights(individual_predictions, validation_mae)
        weighted_avg = np.average(predictions_array, axis=0, weights=weights)
        ensemble_results['weighted_average'] = weighted_avg.tolist()
        
        # Median
        ensemble_results['median'] = np.median(predictions_array, axis=0).tolist()
        
        # Trimmed mean
        ensemble_results['trimmed_mean'] = self._calculate_trimmed_mean(predictions_array)
        
        # Select main ensemble method (use weighted average if validation_mae available, otherwise simple average)
        if validation_mae:
            main_ensemble = ensemble_results['weighted_average']
            method_used = 'weighted_average'
        else:
            main_ensemble = ensemble_results['simple_average']
            method_used = 'simple_average'
        
        return {
            'predictions': main_ensemble,
            'all_methods': ensemble_results,
            'method_used': method_used,
            'weights_used': dict(zip(individual_predictions.keys(), weights)) if validation_mae else None
        }
    
    def _calculate_model_weights(self, individual_predictions: Dict[str, List[float]], validation_mae: Dict[str, float]) -> List[float]:
        """Calculate model weights using LLM based on validation MAE performance"""
        # logger.info("Calculating model weights using LLM...")
        
        if not individual_predictions or not validation_mae:
            # Fallback to uniform weights
            n_models = len(individual_predictions)
            return [1.0 / n_models] * n_models
        
        # Prepare model performance data for LLM
        model_performance = {}
        for model_name in individual_predictions.keys():
            if model_name in validation_mae:
                model_performance[model_name] = {
                    'mae': validation_mae[model_name],
                    'predictions_count': len(individual_predictions[model_name])
                }
        
        if not model_performance:
            logger.warning("No validation MAE data available, using uniform weights")
            n_models = len(individual_predictions)
            return [1.0 / n_models] * n_models
        
        # Create prompt for LLM weight assignment
        prompt = f"""
You are an expert time series forecasting analyst. You need to assign weights to {len(model_performance)} models based on their validation performance.

**Model Performance on Validation Set:**
{json.dumps(model_performance, indent=2)}

**Your Task:**
Analyze the validation MAE (Mean Absolute Error) for each model and assign appropriate weights for ensemble forecasting. Consider:
1. Lower MAE indicates better performance - these models should get higher weights
2. Weights should sum to 1.0
3. No model should get a weight of 0 (unless it completely failed)
4. Consider the relative performance differences between models

**Weight Assignment Guidelines:**
- Models with lower MAE should get higher weights
- The best performing model should get the highest weight
- Weights should be proportional to performance but not necessarily linear
- Consider giving some weight even to poorer performing models for diversity

**Return your decision in JSON format:**
{{
    "weights": {{
        "model_name": float (weight between 0 and 1)
    }},
    "reasoning": "string (explain your weight assignment strategy)",
    "total_weight": float (should be 1.0)
}}

**Example Response:**
{{
    "weights": {{
        "ModelA": 0.4,
        "ModelB": 0.35,
        "ModelC": 0.25
    }},
    "reasoning": "ModelA has the lowest MAE and shows consistent performance, ModelB is second best, ModelC has higher error but still contributes to ensemble diversity.",
    "total_weight": 1.0
}}

IMPORTANT: Ensure the sum of all weights equals exactly 1.0.
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            logger.info(f"LLM response for weight calculation: {response.content}")
            
            # Extract weights from LLM response
            weight_decision = self._extract_weights_from_response(response.content, list(model_performance.keys()))
            
            # Convert to list format in the same order as individual_predictions
            weights_list = []
            for model_name in individual_predictions.keys():
                if model_name in weight_decision:
                    weights_list.append(weight_decision[model_name])
                else:
                    # Fallback weight for models not in validation_mae
                    weights_list.append(0.1)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights_list)
            if total_weight > 0:
                weights_list = [w / total_weight for w in weights_list]
            else:
                # Fallback to uniform weights
                n_models = len(individual_predictions)
                weights_list = [1.0 / n_models] * n_models
            
            logger.info(f"Calculated weights: {dict(zip(individual_predictions.keys(), weights_list))}")
            return weights_list
            
        except Exception as e:
            logger.warning(f"Failed to calculate weights using LLM: {e}, using uniform weights")
            n_models = len(individual_predictions)
            return [1.0 / n_models] * n_models
    
    def _extract_weights_from_response(self, response_content: str, model_names: List[str]) -> Dict[str, float]:
        """Extract weights from LLM response"""
        try:
            # Try to extract JSON from response
            import re
            import json
            
            # Look for JSON pattern in the response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                weight_data = json.loads(json_str)
                
                if 'weights' in weight_data:
                    weights = weight_data['weights']
                    # Validate weights
                    for model_name in model_names:
                        if model_name not in weights:
                            weights[model_name] = 0.1  # Default weight
                        elif weights[model_name] < 0:
                            weights[model_name] = 0.1  # Ensure non-negative
                    
                    return weights
            
            # Fallback: try to extract weights using regex
            weights = {}
            for model_name in model_names:
                # Look for pattern like "ModelA": 0.4 or "ModelA":0.4
                pattern = rf'"{model_name}"\s*:\s*([0-9]*\.?[0-9]+)'
                match = re.search(pattern, response_content)
                if match:
                    weights[model_name] = float(match.group(1))
                else:
                    weights[model_name] = 0.1  # Default weight
            
            return weights
            
        except Exception as e:
            logger.warning(f"Failed to extract weights from response: {e}")
            # Return uniform weights as fallback
            return {model_name: 1.0 / len(model_names) for model_name in model_names}
    
    def _calculate_trimmed_mean(self, predictions_array: np.ndarray, trim_percent: float = 0.1) -> List[float]:
        """Calculate trimmed mean"""
        trimmed_predictions = []
        
        for i in range(predictions_array.shape[1]):
            values = predictions_array[:, i]
            sorted_values = np.sort(values)
            n_trim = int(len(values) * trim_percent)
            trimmed_values = sorted_values[n_trim:-n_trim] if n_trim > 0 else sorted_values
            trimmed_predictions.append(np.mean(trimmed_values))
        
        return trimmed_predictions
    
    def _calculate_forecast_metrics(self, individual_predictions: Dict[str, List[float]],
                                  ensemble_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate prediction metrics"""
        logger.info("Calculating forecast metrics...")
        
        metrics = {}
        
        # Calculate prediction statistics for each model
        for model, predictions in individual_predictions.items():
            metrics[model] = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'range': np.max(predictions) - np.min(predictions)
            }
        
        # Calculate ensemble prediction statistics
        if ensemble_predictions:
            ensemble_pred = ensemble_predictions['predictions']
            metrics['ensemble'] = {
                'mean': np.mean(ensemble_pred),
                'std': np.std(ensemble_pred),
                'min': np.min(ensemble_pred),
                'max': np.max(ensemble_pred),
                'range': np.max(ensemble_pred) - np.min(ensemble_pred)
            }
        
        return metrics
    
    def _calculate_test_metrics(self, individual_predictions: Dict[str, List[float]],
                               ensemble_predictions: Dict[str, Any], test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate test set metrics (MSE, MAE, MAPE) for all models and ensemble"""
        logger.info("Calculating test set metrics...")
        
        actual_values = test_data['value'].values
        test_metrics = {}
        
        # Calculate metrics for each individual model
        for model_name, predictions in individual_predictions.items():
            try:
                # Ensure predictions and actual values have same length
                if len(predictions) != len(actual_values):
                    min_len = min(len(predictions), len(actual_values))
                    pred_values = predictions[:min_len]
                    act_values = actual_values[:min_len]
                else:
                    pred_values = predictions
                    act_values = actual_values
                
                # Calculate metrics
                mse = mean_squared_error(act_values, pred_values)
                mae = mean_absolute_error(act_values, pred_values)
                mape = np.mean(np.abs((act_values - pred_values) / np.where(act_values != 0, act_values, 1))) * 100
                
                test_metrics[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'mape': mape
                }
                
                logger.info(f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                logger.warning(f"Failed to calculate metrics for {model_name}: {e}")
                test_metrics[model_name] = {
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'mape': float('inf')
                }
        
        # Calculate metrics for ensemble prediction
        if ensemble_predictions and 'predictions' in ensemble_predictions:
            try:
                ensemble_pred = ensemble_predictions['predictions']
                
                # Ensure predictions and actual values have same length
                if len(ensemble_pred) != len(actual_values):
                    min_len = min(len(ensemble_pred), len(actual_values))
                    pred_values = ensemble_pred[:min_len]
                    act_values = actual_values[:min_len]
                else:
                    pred_values = ensemble_pred
                    act_values = actual_values
                
                # Calculate metrics
                mse = mean_squared_error(act_values, pred_values)
                mae = mean_absolute_error(act_values, pred_values)
                mape = np.mean(np.abs((act_values - pred_values) / np.where(act_values != 0, act_values, 1))) * 100
                
                test_metrics['ensemble'] = {
                    'mse': mse,
                    'mae': mae,
                    'mape': mape
                }
                
                logger.info(f"Ensemble - MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                logger.warning(f"Failed to calculate ensemble metrics: {e}")
                test_metrics['ensemble'] = {
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'mape': float('inf')
                }
        
        return test_metrics
    
    def _generate_confidence_intervals(self, individual_predictions: Dict[str, List[float]],
                                     ensemble_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence intervals"""
        logger.info("Generating confidence intervals...")
        
        if not individual_predictions:
            return {}
        
        predictions_array = np.array(list(individual_predictions.values()))
        
        confidence_intervals = {}
        
        # Calculate percentiles for confidence intervals
        for confidence_level in [0.8, 0.9, 0.95]:
            lower_percentile = (1 - confidence_level) / 2 * 100
            upper_percentile = (1 + confidence_level) / 2 * 100
            
            lower_bounds = np.percentile(predictions_array, lower_percentile, axis=0)
            upper_bounds = np.percentile(predictions_array, upper_percentile, axis=0)
            
            confidence_intervals[f'{int(confidence_level*100)}%'] = {
                'lower': lower_bounds.tolist(),
                'upper': upper_bounds.tolist()
            }
        
        return confidence_intervals
    
    def _generate_forecast_visualizations(self, validation_data: pd.DataFrame, individual_predictions: Dict[str, Any],
                                       ensemble_predictions: Dict[str, Any], confidence_intervals: Dict[str, Any],
                                       test_data: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """Generate forecast visualizations"""
        logger.info("Generating forecast visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualizations = {}
        
        try:
            # Determine which predictions to use for visualization
            if isinstance(individual_predictions, dict) and 'original_scale' in individual_predictions:
                # Use original scale predictions if available
                viz_individual_predictions = individual_predictions['original_scale']
                viz_ensemble_predictions = ensemble_predictions['original_scale']
            else:
                # Use standardized predictions
                viz_individual_predictions = individual_predictions
                viz_ensemble_predictions = ensemble_predictions
            
            # 1. Forecast comparison plot
            comparison_plot_path = output_path / "forecast_comparison.png"
            visualizations['forecast_comparison'] = self._plot_forecast_comparison(
                validation_data, viz_individual_predictions, viz_ensemble_predictions, test_data, str(comparison_plot_path)
            )
            
            # 2. Ensemble forecast plot
            ensemble_plot_path = output_path / "ensemble_forecast.png"
            visualizations['ensemble_forecast'] = self._plot_ensemble_forecast(
                validation_data, viz_individual_predictions, viz_ensemble_predictions, confidence_intervals, test_data, str(ensemble_plot_path)
            )
            
            # 3. Forecast distribution plot
            distribution_plot_path = output_path / "forecast_distribution.png"
            visualizations['forecast_distribution'] = self._plot_forecast_distribution(
                viz_individual_predictions, str(distribution_plot_path)
            )
            
            logger.info(f"Generated {len(visualizations)} forecast visualizations")
            
        except Exception as e:
            logger.error(f"Forecast visualization generation failed: {e}")
            visualizations = {}
        
        return visualizations
    
    def _plot_forecast_comparison(self, validation_data: pd.DataFrame, individual_predictions: Dict[str, List[float]],
                                ensemble_predictions: Dict[str, Any], test_data: pd.DataFrame, save_path: str) -> str:
        """Plot forecast comparison"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot validation data as historical data
            ax.plot(validation_data.index, validation_data['value'], 'b-', label='Input Data', linewidth=2)
            
            # Plot validation data as actual values
            ax.plot(test_data.index, test_data['value'], 'g-', label='Test Data', linewidth=2)
            test_index = test_data.index
            # Plot individual model predictions
            colors = plt.cm.Set3(np.linspace(0, 1, len(individual_predictions)))
            for i, (model, predictions) in enumerate(individual_predictions.items()):
                # 使用test_data的时间索引，但只取预测长度
                if len(predictions) <= len(test_index):
                    model_forecast_index = test_index[:len(predictions)]
                else:
                    # 如果预测长度超过测试数据长度，扩展时间索引
                    if hasattr(test_index, 'freq') and test_index.freq is not None:
                        last_date = test_index[-1]
                        extended_dates = pd.date_range(start=last_date + test_index.freq, 
                                                    periods=len(predictions) - len(test_index), 
                                                    freq=test_index.freq)
                        model_forecast_index = test_index.union(extended_dates)
                    else:
                        # 如果没有频率信息，使用数值索引
                        extended_index = pd.RangeIndex(start=len(test_index), 
                                                    stop=len(test_index) + len(predictions) - len(test_index))
                        model_forecast_index = test_index.union(extended_index)
                
                ax.plot(model_forecast_index, predictions, '--', color=colors[i], 
                    label=f'{model}', alpha=0.7, linewidth=1.5)
            
            # Plot ensemble prediction
            if ensemble_predictions:
                ensemble_pred = ensemble_predictions['predictions']
                # 使用test_data的时间索引，但只取预测长度
                if len(ensemble_pred) <= len(test_index):
                    ensemble_forecast_index = test_index[:len(ensemble_pred)]
                else:
                    # 如果预测长度超过测试数据长度，扩展时间索引
                    if hasattr(test_index, 'freq') and test_index.freq is not None:
                        last_date = test_index[-1]
                        extended_dates = pd.date_range(start=last_date + test_index.freq, 
                                                    periods=len(ensemble_pred) - len(test_index), 
                                                    freq=test_index.freq)
                        ensemble_forecast_index = test_index.union(extended_dates)
                    else:
                        # 如果没有频率信息，使用数值索引
                        extended_index = pd.RangeIndex(start=len(test_index), 
                                                    stop=len(test_index) + len(ensemble_pred) - len(test_index))
                        ensemble_forecast_index = test_index.union(extended_index)
                
                ax.plot(ensemble_forecast_index, ensemble_pred, 'r-', label='Ensemble', 
                    linewidth=3, alpha=0.9)

            
            ax.set_title('Time Series Forecast Comparison', fontweight='bold', fontsize=14)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Forecast comparison plot failed: {e}")
            return ""
    
    def _plot_ensemble_forecast(self, validation_data: pd.DataFrame, individual_predictions: Dict[str, List[float]],
                              ensemble_predictions: Dict[str, Any], confidence_intervals: Dict[str, Any],
                              test_data: pd.DataFrame, save_path: str) -> str:
        """Plot ensemble forecast with individual model forecasts and confidence intervals"""
        try:
            import matplotlib.pyplot as plt
            
            # Debug: Print data index information
            logger.info(f"Validation data index type: {type(validation_data.index)}")
            logger.info(f"Validation data index range: {validation_data.index.min()} to {validation_data.index.max()}")
            logger.info(f"Validation data index frequency: {getattr(validation_data.index, 'freq', 'None')}")
            
            # Get the last portion of validation data for context
            context_data = validation_data.tail(200)  # Last 200 observations for better context
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot historical data
            ax.plot(context_data.index, context_data['value'], 
                   label='Historical Data', color='black', linewidth=2, alpha=0.8)
            ax.plot(test_data.index, test_data['value'], 'g-', label='Test Data', linewidth=2)
            
            # Create future time index
            last_date = context_data.index[-1]
            if isinstance(last_date, pd.Timestamp):
                if hasattr(validation_data.index, 'freq') and validation_data.index.freq is not None:
                    # If data has a frequency, extend the index properly
                    forecast_dates = pd.date_range(start=last_date + validation_data.index.freq, 
                                                 periods=len(ensemble_predictions['predictions']), 
                                                 freq=validation_data.index.freq)
                else:
                    # Try to infer frequency from the data
                    if len(validation_data.index) > 1:
                        inferred_freq = pd.infer_freq(validation_data.index)
                        if inferred_freq:
                            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                                         periods=len(ensemble_predictions['predictions']), 
                                                         freq=inferred_freq)
                        else:
                            # Use hourly frequency as default
                            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                                         periods=len(ensemble_predictions['predictions']), 
                                                         freq='h')
                    else:
                        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), 
                                                     periods=len(ensemble_predictions['predictions']), 
                                                     freq='h')
            else:
                # If not datetime index, use sequential index
                forecast_dates = pd.RangeIndex(start=len(context_data), 
                                             stop=len(context_data) + len(ensemble_predictions['predictions']))
            
            # Plot individual model forecasts with different colors
            colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            for i, (model_name, predictions) in enumerate(individual_predictions.items()):
                color = colors[i % len(colors)]
                # Use only the available forecast dates for this model
                model_forecast_dates = forecast_dates[:len(predictions)]
                ax.plot(model_forecast_dates, predictions, 
                       label=f'{model_name}', color=color, linewidth=1.5, alpha=0.7, linestyle='--')
            
            # Plot ensemble forecast (highlighted)
            ensemble_pred = ensemble_predictions['predictions']
            ensemble_forecast_dates = forecast_dates[:len(ensemble_pred)]
            ax.plot(ensemble_forecast_dates, ensemble_pred, 
                   label='Ensemble Forecast', color='red', linewidth=3, marker='o', markersize=6)
            
            # Plot confidence intervals
            if confidence_intervals:
                for confidence_level, intervals in confidence_intervals.items():
                    lower = intervals['lower']
                    upper = intervals['upper']
                    # Use only the available forecast dates for confidence intervals
                    ci_forecast_dates = forecast_dates[:len(lower)]
                    ax.fill_between(ci_forecast_dates, lower, upper, alpha=0.2, 
                                  label=f'{confidence_level} Confidence')
            
            # Add vertical line to separate historical and forecast
            ax.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
            
            # Customize the plot
            ax.set_title('Ensemble Forecast with Individual Model Predictions', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add ensemble method information
            ensemble_method = ensemble_predictions.get('method_used', 'unknown')
            method_text = f'Ensemble Method: {ensemble_method}'
            ax.text(0.02, 0.98, method_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save plot
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Ensemble forecast plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Ensemble forecast plot failed: {e}")
            return ""
    
    def _plot_forecast_distribution(self, individual_predictions: Dict[str, List[float]], save_path: str) -> str:
        """Plot forecast distribution"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            # Plot distribution for each prediction step
            for i in range(min(4, len(list(individual_predictions.values())[0]))):
                values = [predictions[i] for predictions in individual_predictions.values()]
                
                axes[i].hist(values, bins=10, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Step {i+1} Distribution')
                axes[i].set_xlabel('Predicted Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle('Forecast Distribution by Step', fontweight='bold', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Forecast distribution plot failed: {e}")
            return ""
    
    def _save_forecast_results(self, individual_predictions: Dict[str, Any],
                             ensemble_predictions: Dict[str, Any], forecast_metrics: Dict[str, Any],
                             confidence_intervals: Dict[str, Any], output_dir: str):
        """Save forecast results"""
        logger.info("Saving forecast results...")
        
        output_path = Path(output_dir)
        
        # Save forecast report
        from utils.file_utils import FileSaver
        forecast_report = {
            'individual_predictions': individual_predictions,
            'ensemble_predictions': ensemble_predictions,
            'forecast_metrics': forecast_metrics,
            'confidence_intervals': confidence_intervals
        }
        
        report_path = output_path / "forecast_report.json"
        FileSaver.save_json(forecast_report, report_path)
        logger.info(f"Forecast report saved to {report_path}")
    
    def _update_memory(self, individual_predictions: Dict[str, Any],
                      ensemble_predictions: Dict[str, Any], forecast_metrics: Dict[str, Any],
                      confidence_intervals: Dict[str, Any], visualizations: Dict[str, str]):
        """Update memory"""
        self.memory.store('individual_predictions', individual_predictions, 'forecasts')
        self.memory.store('ensemble_predictions', ensemble_predictions, 'forecasts')
        self.memory.store('forecast_metrics', forecast_metrics, 'forecasts')
        self.memory.store('confidence_intervals', confidence_intervals, 'forecasts')
        self.memory.store('forecast_visualizations', visualizations, 'visualizations')
        
        # Record forecast history
        self.memory.add_history(
            'forecast',
            {
                'models_count': len(individual_predictions),
                'ensemble_method': ensemble_predictions.get('method_used', 'unknown'),
                'visualization_count': len(visualizations)
            }
        )
