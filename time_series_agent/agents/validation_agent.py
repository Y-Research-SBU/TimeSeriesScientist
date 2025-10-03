"""
Validation Agent for Time Series Prediction
Validation Agent - responsible for model selection, hyperparameter optimization, and validation set evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from typing_extensions import Annotated, TypedDict
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import json
from itertools import product

from agents.memory import ExperimentMemory
from utils.visualization_utils import TimeSeriesVisualizer
from utils.model_library import get_model_function
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TypedDict for model selection output
class SelectedModel(TypedDict):
    """A selected model with its hyperparameters and reasoning."""
    model: Annotated[str, ..., "Name of the selected model"]
    hyperparameters: Annotated[Dict[str, Any], ..., "Hyperparameters for the model"]
    reason: Annotated[str, ..., "Reason for selecting this model"]

class ModelSelectionOutput(TypedDict):
    """Output format for model selection."""
    selected_models: Annotated[List[SelectedModel], ..., "List of selected models with hyperparameters"]

VALIDATION_SYSTEM_PROMPT = """
You are the Model Selection and Validation Lead Agent for an industrial time series forecasting system.

Background:
- You are highly skilled in matching data characteristics to appropriate forecasting models and in designing robust validation strategies.
- You understand the strengths, weaknesses, and requirements of a wide range of statistical and machine learning models.

Your responsibilities:
- Review the data analysis summary and select the top 3 most suitable forecasting models from the provided list.
- For each model, recommend a hyperparameter search space tailored to the data's characteristics and modeling goals.
- Justify each model choice and hyperparameter range, referencing both the analysis and your domain expertise.
- Consider diversity in model selection to maximize ensemble robustness.
- Always return your decisions in a structured Python dict, with clear reasoning for each choice.

You have access to:
- The data analysis summary (as a Python dict)
- The list of available models

Your output will directly determine which models are trained and how they are tuned.
"""

def get_model_selection_prompt(analysis: dict, available_models: list, n_candidates: int) -> str:
    prompt = f"""
You are a time series model selection agent. Given the analysis report {analysis} and available models {available_models}, select the best {n_candidates} models that are most suitable for the data and propose hyperparameters for each model.

For each model, you should propose a hyperparameter search space tailored to the data's characteristics and modeling goals.
Justify each model choice and hyperparameter range, referencing both the analysis and your domain expertise.

Return your answer in the following JSON format with an array of selected models:

{{
    "selected_models": [
        {{
            "model": "string",
            "hyperparameters": {{...}},
            "reason": "string"
        }},
        {{
            "model": "string",
            "hyperparameters": {{...}},
            "reason": "string"
        }},
    ]
}}

Below is an example of the output:

{{
    "selected_models": [
        {{
            "model": "ARIMA",
            "hyperparameters": {{
                "p": [0, 1, 2],
                "d": [0, 1],
                "q": [0, 1, 2],
            }},
            "reason": "string"
        }},
    ]
}}

IMPORTANT REQUIREMENTS:
1. Return EXACTLY {n_candidates} models in the selected_models array
2. Each model must have "model", "hyperparameters", and "reason" fields
3. The "model" field must be one of the available models: {available_models}
4. The "hyperparameters" field should contain 2-3 parameter search spaces as arrays
5. Return ONLY the JSON object, no markdown formatting, no explanations before or after
6. Ensure the JSON is valid and properly formatted
"""
    return prompt

class ValidationAgent:
    """
    Validation Agent
    Responsible for model selection, hyperparameter optimization, and validation set evaluation.
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
        self.available_models = self.config.get('available_models')
        self.n_candidates = self.config.get('n_candidates', 5)
        self.cv_folds = self.config.get('cv_folds', 5)
        self.k_models = self.config.get('k_models', 3)
        self.optimization_method = self.config.get('optimization_method', 'grid_search')

    def run(self, analysis_result: str, available_models: List[str], validation_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run the validation agent to select and validate models"""
        import time
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        logger.info(f"Starting model validation with {len(available_models)} available models")
        
        try:
            # Parse LLM response to get model selection
            model_selection = self._parse_structured_response(analysis_result, available_models)
            
            if not model_selection or not model_selection.get('selected_models'):
                logger.warning("No models selected by LLM, using fallback selection")
                model_selection = {
                    'selected_models': available_models[:self.config.get('k_models', 3)],
                    'reasoning': 'Fallback selection due to LLM failure'
                }
                
            selected_models = model_selection['selected_models']
            logger.info(f"LLM selected {len(selected_models)} models: {selected_models}")

            tested_models = self._test_models_on_validation_data(selected_models, validation_data)
            
            if not tested_models:
                logger.warning("No models successfully tested, using fallback models")
                tested_models = self._generate_fallback_models(available_models)
            
            # Select best models from testing results
            best_models = self._select_best_models_from_testing(tested_models)
            
            logger.info(f"Validation completed. Selected {len(best_models)} best models")
            return best_models
            
        except Exception as e:
            logger.error(f"Error in validation agent: {e}")
            logger.info("Using fallback model selection")
            return self._generate_fallback_models(available_models)
    
    def _parse_structured_response(self, analysis_result: str, available_models: List[str]) -> Dict[str, Any]:
        """Parse LLM response to get model selection"""
        try:
            # Create structured output LLM
            structured_llm = self.llm.with_structured_output(ModelSelectionOutput)
            
            # Convert analysis_result to dict if it's a string
            if isinstance(analysis_result, str):
                # Try to parse as JSON first
                try:
                    import json
                    analysis_dict = json.loads(analysis_result)
                except (json.JSONDecodeError, ValueError):
                    # If not valid JSON, create a simple dict with the string as summary
                    analysis_dict = {
                        'summary': analysis_result,
                        'trend_analysis': 'Analysis provided as text',
                        'seasonality_analysis': 'Analysis provided as text',
                        'stationarity': 'Analysis provided as text',
                        'potential_issues': 'Analysis provided as text'
                    }
            else:
                analysis_dict = analysis_result
            
            # Create prompt for model selection
            prompt = get_model_selection_prompt(analysis_dict, available_models, self.n_candidates)
            
            # Invoke the structured LLM
            selected_models_info = structured_llm.invoke([
                SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            logger.info("Successfully parsed LLM response with structured output")
            return selected_models_info
            
        except Exception as e:
            logger.error(f"Error parsing structured response: {e}")
            return None
    
    def _test_models_on_validation_data(self, selected_models: List[str], validation_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Test selected models on validation data"""
        tested_models = []
        
        for model_info in selected_models:
            model_name = model_info['model']
            hyperparameters = model_info['hyperparameters']
            try:
                logger.info(f"Testing model: {model_name}")
                
                # Get default hyperparameters for this model
                # default_hyperparams = self._get_default_hyperparameters(model_name)
                
                # Optimize hyperparameters for this model
                best_hyperparams, best_metrics = self._optimize_model_hyperparameters(validation_data, model_name, hyperparameters)
                
                tested_models.append({
                    'model': model_name,
                    'hyperparameters': best_hyperparams,
                    'validation_score': best_metrics['mse'],
                    'validation_metrics': best_metrics
                })
                
                logger.info(f"Model {model_name} tested successfully with score: {best_metrics['mse']:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to test model {model_name}: {e}")
                # Add with infinite score so it won't be selected
                tested_models.append({
                    'model': model_name,
                    'hyperparameters': {},
                    'validation_score': float('inf'),
                    'validation_metrics': {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
                })
        
        return tested_models
    
    def _generate_fallback_models(self, available_models: List[str]) -> List[Dict[str, Any]]:
        """Generate fallback models when validation fails"""
        logger.info("Generating fallback models...")
        
        fallback_models = []
        k_models = min(self.config.get('k_models', 3), len(available_models))
        
        for i in range(k_models):
            model_name = available_models[i]
            fallback_models.append({
                'model': model_name,
                'hyperparameters': {},
                'validation_score': 1.0 + i * 0.1,  # Simple fallback scores
                'validation_metrics': {
                    'mse': 1.0 + i * 0.1,
                    'mae': 0.8 + i * 0.08,
                    'mape': 20.0 + i * 2.0
                }
            })
        
        return fallback_models
    

    
    def _optimize_model_hyperparameters(self, validation_data: pd.DataFrame, 
                                      model_name: str, hyperparameters: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Optimize hyperparameters for a specific model using grid search on validation set.
        
        Args:
            validation_data: Validation dataset
            model_name: Name of the model
            hyperparameters: Model hyperparameters (dict with lists of values)
            
        Returns:
            Tuple of (best_hyperparameters, best_metrics)
        """
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        if not hyperparameters:
            logger.info(f"No hyperparameters to optimize for {model_name}, using defaults")
            # Use default parameters and evaluate
            default_metrics = self._evaluate_model_on_validation(validation_data, model_name, {})
            return {}, default_metrics
        
        # Generate parameter combinations (grid search)
        param_names = list(hyperparameters.keys())
        param_values = list(hyperparameters.values())
        
        # Limit the number of combinations to avoid excessive computation
        max_combinations = 20
        combinations = []
        
        # Generate combinations using itertools
        for combination in product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
            if len(combinations) >= max_combinations:
                break
        
        logger.info(f"Testing {len(combinations)} hyperparameter combinations...")
        
        best_params = {}
        best_metrics = {'mse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}
        
        # Test each combination
        for i, params in enumerate(combinations):
            try:
                # Evaluate model with these parameters
                metrics = self._evaluate_model_on_validation(validation_data, model_name, params)
                
                if metrics['mae'] < best_metrics['mae']:
                    best_metrics = metrics
                    best_params = params.copy()
                
                print(f"Combination {i+1}/{len(combinations)}: MSE = {metrics['mse']:.4f}, MAE = {metrics['mae']:.4f}, MAPE = {metrics['mape']:.2f}%, Params = {params}")
                
            except Exception as e:
                logger.info(f"Combination {i+1}/{len(combinations)} failed: {e}")
                continue
        
        print(f"Best hyperparameters for {model_name}: {best_params} (MAE = {best_metrics['mae']:.4f})")
        return best_params, best_metrics
    
    def _evaluate_model_on_validation(self, validation_data: pd.DataFrame, 
                                    model_name: str, hyperparameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single model with specific hyperparameters on validation data
        
        Args:
            validation_data: Validation dataset
            model_name: Name of the model
            hyperparameters: Model hyperparameters (single parameter set, not list)
            
        Returns:
            Dictionary with MSE, MAE, MAPE scores
        """
        try:
            # Split validation data
            split_point = int(len(validation_data) * 0.8)
            train_data = validation_data.iloc[:split_point]
            test_data = validation_data.iloc[split_point:]
                        
            predictions = self._validate_and_predict(train_data, model_name, hyperparameters, len(test_data))
            
            # Calculate metrics on the test portion (20% of validation data)
            actual_values = test_data['value'].values
            
            # Ensure predictions and actual values have same length
            if len(predictions) != len(actual_values):
                logger.warning(f"Predictions and actual values have different lengths: {len(predictions)} != {len(actual_values)}")
                min_len = min(len(predictions), len(actual_values))
                predictions = predictions[:min_len]
                actual_values = actual_values[:min_len]
            
            # Calculate metrics
            mse = mean_squared_error(actual_values, predictions)
            mae = mean_absolute_error(actual_values, predictions)
            mape = np.mean(np.abs((actual_values - predictions) / np.where(actual_values != 0, actual_values, 1))) * 100
            
            return {
                'mse': mse,
                'mae': mae,
                'mape': mape
            }
            
        except Exception as e:
            logger.warning(f"Model evaluation failed for {model_name}: {e}")
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'mape': float('inf')
            }
    
    def _validate_and_predict(self, validation_data: pd.DataFrame,
                          model_name: str, hyperparameters: Dict[str, Any], horizon: int = None) -> List[float]:
        """
        Train a model on validation data and make predictions with specific hyperparameters
        
        Args:
            validation_data: Validation dataset
            model_name: Name of the model
            hyperparameters: Model hyperparameters (single parameter set)
            horizon: Number of steps to predict (if None, predicts for entire validation data length)
            
        Returns:
            List of predictions
        """
        try:
            # Prepare data
            values = validation_data['value'].values.reshape(-1, 1)
            
            # Create features for the entire validation data
            actual_values = values.flatten()
            
            # Train model with specific parameters
            predictions = self._train_single_model(model_name, actual_values, hyperparameters, horizon)
            
            if predictions is not None:
                return predictions.tolist()
            else:
                # Fallback to default parameters
                logger.warning(f"Model training failed for {model_name}, using fallback")
                fallback_predictions = self._train_single_model(model_name, actual_values, {}, horizon)
                if fallback_predictions is not None:
                    return fallback_predictions.tolist()
                else:
                    # Final fallback to simple predictions
                    logger.warning(f"Fallback model training also failed for {model_name}, using simple predictions")
                    seed = hash(str(hyperparameters)) % 10000
                    np.random.seed(seed)
                    predictions = np.linspace(0.3, 0.7, horizon) + np.random.normal(0, 0.1, horizon)
                    predictions = np.clip(predictions, 0, 1)
                    return predictions.tolist()
                
        except Exception as e:
            logger.warning(f"Model training failed for {model_name}: {e}")
            # Fallback to simple predictions
            n_predictions = len(validation_data)
            seed = hash(str(hyperparameters)) % 10000
            np.random.seed(seed)
            predictions = np.linspace(0.3, 0.7, n_predictions) + np.random.normal(0, 0.1, n_predictions)
            predictions = np.clip(predictions, 0, 1)
            return predictions.tolist()
    
    def _train_single_model(self, model_name: str,
                           actual_values: np.ndarray, params: Dict[str, Any], horizon: int) -> np.ndarray:
        """
        Train a single model with specific parameters using model library
        
        Args:
            model_name: Name of the model
            features: Feature matrix
            actual_values: Target values
            params: Model parameters
            horizon: Number of steps to predict
            
        Returns:
            Predictions array
        """
        try:
            # Get the model function from the library
            model_func = get_model_function(model_name)
            
            # Prepare data in the format expected by model library functions
            data_dict = {'value': actual_values}
            
            # Call the model function with the data, parameters, and horizon
            predictions = model_func(data_dict, params, horizon)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.debug(f"Single model training failed: {e}")
            return None
    
    def _select_best_models_from_testing(self, tested_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select the best models based on validation performance
        
        Args:
            tested_models: List of tested models with scores
            
        Returns:
            List of best models (top k_models)
        """
        # Protect against None input

        
        if not tested_models:
            logger.warning("tested_models is empty in _select_best_models_from_testing")
            return []
        
        # Sort by validation score (lower is better)
        sorted_models = sorted(tested_models, key=lambda x: x['validation_score'])
        
        # Select top k_models
        best_models = sorted_models[:self.k_models]
        
        logger.info(f"Selected best models: {[m['model'] for m in best_models]}")
        for model in best_models:
            metrics = model['validation_metrics']
            logger.info(f"  {model['model']}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
        
        return best_models
    
    def _calculate_model_suitability_score(self, model: str, analysis_result: Dict[str, Any]) -> float:
        """Calculate model suitability score"""
        score = 0.0
        
        data_characteristics = analysis_result.get('data_characteristics', {})
        
        for col, char in data_characteristics.items():
            data_type = char.get('data_type', 'unknown')
            has_trend = char.get('has_trend', False)
            has_seasonality = char.get('has_seasonality', False)
            is_stationary = char.get('is_stationary', False)
            
            # Score based on model characteristics and data characteristics
            if model == 'ARMA' and is_stationary:
                score += 10
            elif model == 'ARIMA' and has_trend:
                score += 8
            elif model == 'SARIMA' and has_seasonality:
                score += 9
            elif model == 'Prophet' and (has_trend or has_seasonality):
                score += 8
            elif model == 'LSTM':
                score += 6  # High versatility
            elif model == 'RandomForest':
                score += 5
            elif model == 'LinearRegression' and has_trend:
                score += 7
            elif model == 'ExponentialSmoothing' and has_seasonality:
                score += 7
            else:
                score += 3  # Base score
        
        return score
    
    def _get_default_hyperparameters(self, model: str) -> Dict[str, Any]:
        """Get default hyperparameters"""
        defaults = {
            'ARMA': {'p': 1, 'q': 1, 'd': 0},
            'ARIMA': {'p': 1, 'q': 1, 'd': 1},
            'SARIMA': {'p': 1, 'q': 1, 'd': 1, 'P': 1, 'Q': 1, 'D': 1, 's': 7},
            'LSTM': {'units': 50, 'layers': 1, 'dropout': 0.1, 'epochs': 50},
            'RandomForest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            'LinearRegression': {},
            'SVR': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
            'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            'Prophet': {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1.0, 'yearly_seasonality': True,'weekly_seasonality': True,'daily_seasonality': False,'seasonality_mode': 'additive'},
            'ExponentialSmoothing': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7}
        }
        return defaults.get(model, {})
    
    def _grid_search_optimization(self, data: pd.DataFrame, model: str, param_grid: Dict[str, List]) -> Tuple[Dict, float]:
        """Perform grid search optimization"""
        best_score = float('inf')
        best_params = {}
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Limit search combinations
        max_combinations = 20
        if len(param_combinations) > max_combinations:
            import random
            param_combinations = random.sample(param_combinations, max_combinations)
        
        for params in param_combinations:
            try:
                score = self._evaluate_model(data, model, params)
                if score < best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.debug(f"Parameter combination {params} failed: {e}")
                continue
        
        return best_params, best_score
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate parameter combinations"""
        import itertools
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def _evaluate_model(self, data: pd.DataFrame, model: str, params: Dict[str, Any]) -> float:
        """Evaluate model performance"""
        try:
            # This should call the actual model training and evaluation
            # For now, return a random score as an example
            return np.random.uniform(0.1, 10.0)
        except Exception as e:
            logger.debug(f"Model evaluation failed: {e}")
            return float('inf')
    
    def _evaluate_model_split(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                            model: str, params: Dict[str, Any]) -> float:
        """Evaluate a single data split"""
        try:
            # This should implement actual model training and prediction
            # For now, return a random score
            return np.random.uniform(0.1, 10.0)
        except Exception as e:
            logger.debug(f"Model split evaluation failed: {e}")
            return float('inf')
    
    def _select_best_models(self, cv_results: Dict[str, Any]) -> List[str]:
        """Select best models"""
        logger.info("Selecting best models...")
        
        # Sort by average score
        model_scores = []
        for model, result in cv_results.items():
            model_scores.append((model, result['mean_score']))
        
        # Select the k_models with the lowest scores
        model_scores.sort(key=lambda x: x[1])
        best_models = [model for model, score in model_scores[:self.k_models]]
        
        logger.info(f"Selected best models: {best_models}")
        return best_models
    
    def _plot_model_performance(self, cv_results: Dict[str, Any], save_path: str) -> str:
        """Plot model performance comparison"""
        try:
            import matplotlib.pyplot as plt
            
            models = list(cv_results.keys())
            mean_scores = [cv_results[model]['mean_score'] for model in models]
            std_scores = [cv_results[model]['std_score'] for model in models]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bars = ax.bar(models, mean_scores, yerr=std_scores, capsize=5, alpha=0.7)
            ax.set_title('Model Performance Comparison (Cross-Validation)', fontweight='bold')
            ax.set_xlabel('Model')
            ax.set_ylabel('Mean Squared Error')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, mean_scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Model performance plot failed: {e}")
            return ""
    
    def _plot_cross_validation(self, cv_results: Dict[str, Any], save_path: str) -> str:
        """Plot cross-validation results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for model, result in cv_results.items():
                cv_scores = result['cv_scores']
                fold_numbers = range(1, len(cv_scores) + 1)
                ax.plot(fold_numbers, cv_scores, 'o-', label=model, alpha=0.7)
            
            ax.set_title('Cross-Validation Scores by Fold', fontweight='bold')
            ax.set_xlabel('Fold Number')
            ax.set_ylabel('Score (MSE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Cross-validation plot failed: {e}")
            return ""
