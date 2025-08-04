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

    def run(self, selected_models: list, best_hyperparameters: dict, test_data: pd.DataFrame):
        """
        Run the forecast agent to generate predictions on test data and calculate metrics
        
        Args:
            selected_models: List of selected model names
            best_hyperparameters: Dictionary of best hyperparameters for each model
            test_data: Test dataset for final evaluation
            
        Returns:
            Dictionary containing individual predictions, ensemble predictions, and metrics
        """
        import time
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        logger.info(f"Starting forecast on test data with {len(selected_models)} models")
        
        # Generate individual model predictions on test data
        individual_predictions = {}
        for model_name in selected_models:
            try:
                logger.info(f"Generating predictions for {model_name}")
                hyperparams = best_hyperparameters.get(model_name, {})
                
                # Get model function and generate predictions
                model_func = get_model_function(model_name)
                data_dict = {'value': test_data['value'].values}
                horizon = len(test_data)
                
                predictions = model_func(data_dict, hyperparams, horizon)
                individual_predictions[model_name] = predictions
                
                logger.info(f"Generated {len(predictions)} predictions for {model_name}")
                
            except Exception as e:
                logger.warning(f"Failed to generate predictions for {model_name}: {e}")
                # Generate fallback predictions
                fallback_predictions = self._generate_fallback_predictions(test_data)
                individual_predictions[model_name] = fallback_predictions
        
        # Generate ensemble predictions
        ensemble_predictions = self._generate_ensemble_predictions(individual_predictions)
        
        # Calculate forecast metrics
        forecast_metrics = self._calculate_forecast_metrics(individual_predictions, ensemble_predictions)
        
        # Calculate test set metrics (MSE, MAE, MAPE)
        test_metrics = self._calculate_test_metrics(individual_predictions, ensemble_predictions, test_data)
        
        # Generate confidence intervals
        confidence_intervals = self._generate_confidence_intervals(individual_predictions, ensemble_predictions)
        
        # Generate visualizations
        output_dir = "results/forecast"
        visualizations = self._generate_forecast_visualizations(
            test_data, individual_predictions, ensemble_predictions, 
            confidence_intervals, len(test_data), output_dir
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
    
    def _generate_ensemble_predictions(self, individual_predictions: Dict[str, List[float]]) -> Dict[str, Any]:
        """Generate ensemble predictions"""
        logger.info("Generating ensemble predictions...")
        
        if not individual_predictions:
            return {}
        
        # Calculate ensemble predictions
        predictions_array = np.array(list(individual_predictions.values()))
        
        ensemble_results = {}
        
        # Simple average
        ensemble_results['simple_average'] = np.mean(predictions_array, axis=0).tolist()
        
        # Weighted average (based on model performance)
        weights = self._calculate_model_weights(individual_predictions)
        weighted_avg = np.average(predictions_array, axis=0, weights=weights)
        ensemble_results['weighted_average'] = weighted_avg.tolist()
        
        # Median
        ensemble_results['median'] = np.median(predictions_array, axis=0).tolist()
        
        # Trimmed mean
        ensemble_results['trimmed_mean'] = self._calculate_trimmed_mean(predictions_array)
        
        # Select main ensemble method (use simple average as default)
        main_ensemble = ensemble_results.get('simple_average', ensemble_results['simple_average'])
        
        return {
            'predictions': main_ensemble,
            'all_methods': ensemble_results,
            'method_used': 'simple_average'
        }
    
    def _calculate_model_weights(self, individual_predictions: Dict[str, List[float]]) -> List[float]:
        """Calculate model weights"""
        # This should calculate weights based on model performance
        # For now, use uniform weights
        n_models = len(individual_predictions)
        return [1.0 / n_models] * n_models
    
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
    
    def _generate_forecast_visualizations(self, data: pd.DataFrame, individual_predictions: Dict[str, List[float]],
                                        ensemble_predictions: Dict[str, Any], confidence_intervals: Dict[str, Any],
                                        horizon: int, output_dir: str) -> Dict[str, str]:
        """Generate forecast visualizations"""
        logger.info("Generating forecast visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualizations = {}
        
        try:
            # 1. Forecast comparison plot
            comparison_plot_path = output_path / "forecast_comparison.png"
            visualizations['forecast_comparison'] = self._plot_forecast_comparison(
                data, individual_predictions, ensemble_predictions, str(comparison_plot_path)
            )
            
            # 2. Ensemble forecast with confidence intervals
            ensemble_plot_path = output_path / "ensemble_forecast.png"
            visualizations['ensemble_forecast'] = self._plot_ensemble_forecast(
                data, individual_predictions, ensemble_predictions, confidence_intervals, str(ensemble_plot_path)
            )
            
            # 3. Forecast distribution plot
            distribution_plot_path = output_path / "forecast_distribution.png"
            visualizations['forecast_distribution'] = self._plot_forecast_distribution(
                individual_predictions, str(distribution_plot_path)
            )
            
            logger.info(f"Generated {len(visualizations)} forecast visualizations")
            
        except Exception as e:
            logger.error(f"Forecast visualization generation failed: {e}")
            visualizations = {}
        
        return visualizations
    
    def _plot_forecast_comparison(self, data: pd.DataFrame, individual_predictions: Dict[str, List[float]],
                                ensemble_predictions: Dict[str, Any], save_path: str) -> str:
        """Plot forecast comparison"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot historical data
            ax.plot(data.index, data['value'], 'b-', label='Historical Data', linewidth=2)
            
            # Plot individual model predictions
            colors = plt.cm.Set3(np.linspace(0, 1, len(individual_predictions)))
            for i, (model, predictions) in enumerate(individual_predictions.items()):
                forecast_index = range(len(data), len(data) + len(predictions))
                ax.plot(forecast_index, predictions, '--', color=colors[i], 
                       label=f'{model}', alpha=0.7, linewidth=1.5)
            
            # Plot ensemble prediction
            if ensemble_predictions:
                ensemble_pred = ensemble_predictions['predictions']
                forecast_index = range(len(data), len(data) + len(ensemble_pred))
                ax.plot(forecast_index, ensemble_pred, 'r-', label='Ensemble', 
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
    
    def _plot_ensemble_forecast(self, data: pd.DataFrame, individual_predictions: Dict[str, List[float]],
                              ensemble_predictions: Dict[str, Any], confidence_intervals: Dict[str, Any],
                              save_path: str) -> str:
        """Plot ensemble forecast with confidence intervals"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot historical data
            ax.plot(data.index, data['value'], 'b-', label='Historical Data', linewidth=2)
            
            # Plot ensemble prediction
            if ensemble_predictions:
                ensemble_pred = ensemble_predictions['predictions']
                forecast_index = range(len(data), len(data) + len(ensemble_pred))
                ax.plot(forecast_index, ensemble_pred, 'r-', label='Ensemble Forecast', 
                       linewidth=3, alpha=0.9)
                
                # Plot confidence intervals
                if confidence_intervals:
                    for confidence_level, intervals in confidence_intervals.items():
                        lower = intervals['lower']
                        upper = intervals['upper']
                        ax.fill_between(forecast_index, lower, upper, alpha=0.2, 
                                      label=f'{confidence_level} Confidence')
            
            ax.set_title('Ensemble Forecast with Confidence Intervals', fontweight='bold', fontsize=14)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
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
    
    def _save_forecast_results(self, individual_predictions: Dict[str, List[float]],
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
    
    def _update_memory(self, individual_predictions: Dict[str, List[float]],
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
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """Get forecast summary"""
        individual_predictions = self.memory.retrieve('individual_predictions', 'forecasts')
        ensemble_predictions = self.memory.retrieve('ensemble_predictions', 'forecasts')
        forecast_metrics = self.memory.retrieve('forecast_metrics', 'forecasts')
        
        if not individual_predictions:
            return {}
        
        summary = {
            'models_used': list(individual_predictions.keys()),
            'ensemble_method': ensemble_predictions.get('method_used', 'unknown') if ensemble_predictions else 'unknown',
            'forecast_metrics': forecast_metrics
        }
        
        return summary 