"""
AgentGraph for Time Series Prediction Agent
Based on LangGraph's workflow for time series prediction
"""

import os
import pandas as pd
import logging
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
import numpy as np

logger = logging.getLogger(__name__)
from agents.preprocess_agent import PreprocessAgent
from agents.analysis_agent import AnalysisAgent
from agents.validation_agent import ValidationAgent
from agents.forecast_agent import ForecastAgent
from agents.report_agent import ReportAgent
from utils.data_utils import DataLoader, DataSplitter, DataPreprocessor
from utils.file_utils import FileManager

class TimeSeriesAgentGraph:
    """
    LangGraph-based orchestrator for time series forecasting agents.
    Each agent is a node; this class only manages orchestration and state transitions.
    """
    def __init__(self, config: Dict[str, Any], model: str = "gpt-4o", debug: bool = False):
        self.config = config
        self.model = model
        self.debug = debug
        self.file_manager = FileManager(config.get('output_dir', 'results'))
        self.path_manager = self.file_manager.path_manager
        # Instantiate agents (API key is read from environment by each agent)
        self.preprocess_agent = PreprocessAgent(model, config)
        self.analysis_agent = AnalysisAgent(model, config)
        self.validation_agent = ValidationAgent(model, config)
        self.forecast_agent = ForecastAgent(model, config)
        self.report_agent = ReportAgent(model, config)
        # Build LangGraph workflow
        self.graph = self._build_graph()

    def _create_agent_nodes(self):
        return {
            "preprocess": self._preprocess_node,
            "analyze": self._analyze_node,
            "validate": self._validate_node,
            "forecast": self._forecast_node,
            "report": self._report_node,
        }

    def _preprocess_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = self.preprocess_agent.run(state["validation_data"])
        state["preprocessed_data"] = result if isinstance(result, pd.DataFrame) else result.get("cleaned_data", state["validation_data"])
        state["preprocess_result"] = result
        return state

    def _analyze_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        visualizations = state["preprocess_result"]["visualizations"]
        result = self.analysis_agent.run(state["preprocessed_data"], visualizations)
        state["analysis_result"] = result
        print("analysis_result: ", state["analysis_result"])
        return state

    def _validate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        available_models = self.config.get('models').get('available_models')
        print(f"{len(available_models)} available models: {available_models}")
        
        # Pass validation data to validation agent
        validation_data = state["preprocessed_data"]
        result = self.validation_agent.run(state["analysis_result"], available_models, validation_data)
        state["validation_result"] = result
        
        # Extract selected models and hyperparameters from result
        # Result is a list of model dicts with validation scores
        state["selected_models"] = [m['model'] for m in result]
        state["best_hyperparameters"] = {m['model']: m['hyperparameters'] for m in result}
        # Also store validation scores for reference
        state["model_validation_scores"] = {m['model']: m['validation_score'] for m in result}
        
        return state

    def _forecast_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Pass selected models, best hyperparameters, and test data to forecast agent
        print(f"Forecast node: Processing slice {state.get('slice_info', {}).get('slice_id', 'unknown')}")
        print(f"Forecast node: Selected models: {state.get('selected_models', [])}")
        print(f"Forecast node: Test data shape: {state.get('test_data', pd.DataFrame()).shape}")
        
        result = self.forecast_agent.run(
            state["selected_models"], 
            state["best_hyperparameters"], 
            state["test_data"]
        )
        
        # print(f"Forecast node: Result keys: {list(result.keys()) if result else 'None'}")
        # if result:
        #     print(f"Forecast node: Individual predictions: {len(result.get('individual_predictions', {}))} models")
        #     print(f"Forecast node: Ensemble predictions: {'Yes' if result.get('ensemble_predictions') else 'No'}")
        #     print(f"Forecast node: Test metrics: {len(result.get('test_metrics', {}))} models")
        
        state["forecast_result"] = result
        return state

    def _report_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Report node: Processing slice {state.get('slice_info', {}).get('slice_id', 'unknown')+1}")
        print(f"Report node: State keys before processing: {list(state.keys())}")
        print(f"Report node: forecast_result exists: {'forecast_result' in state}")
        if 'forecast_result' in state:
            forecast_result = state['forecast_result']
            print(f"Report node: forecast_result keys: {list(forecast_result.keys()) if forecast_result else 'None'}")
        
        # Create experiment summary from state
        experiment_summary = {
            'slice_info': state.get('slice_info', {}),
            'preprocess_result': {
                'cleaned_data_shape': state.get('preprocessed_data', pd.DataFrame()).shape if state.get('preprocessed_data') is not None else None,
                'analysis_report': state.get('preprocess_result', {}).get('analysis_report', {}),
                'visualizations': state.get('preprocess_result', {}).get('visualizations', {}),
                'outlier_info': state.get('preprocess_result', {}).get('outlier_info', {}),
                'preprocess_config': state.get('preprocess_result', {}).get('preprocess_config', {})
            },
            'analysis_result': state.get('analysis_result', {}),
            'validation_result': {
                'selected_models': state.get('selected_models', []),
                'best_hyperparameters': state.get('best_hyperparameters', {}),
                'model_validation_scores': state.get('model_validation_scores', {})
            },
            'forecast_result': {
                'individual_predictions': state.get('forecast_result', {}).get('individual_predictions', {}),
                'ensemble_predictions': state.get('forecast_result', {}).get('ensemble_predictions', {}),
                'test_metrics': state.get('forecast_result', {}).get('test_metrics', {}),
                'forecast_metrics': state.get('forecast_result', {}).get('forecast_metrics', {}),
                'confidence_intervals': state.get('forecast_result', {}).get('confidence_intervals', {}),
                'visualizations': state.get('forecast_result', {}).get('visualizations', {})
            },
            'config': state.get('config', {})
        }
        
        report = self.report_agent.run(experiment_summary)
        state["report"] = report
        
        # IMPORTANT: Keep forecast_result in state for aggregation
        # The forecast_result should already be in state from _forecast_node
        # We don't need to modify it here, just ensure it's preserved
        
        print(f"Report node: State keys after processing: {list(state.keys())}")
        print(f"Report node: forecast_result still exists: {'forecast_result' in state}")
        
        return state

    def _build_graph(self):
        nodes = self._create_agent_nodes()
        workflow = StateGraph(dict)
        workflow.add_node("preprocess", nodes["preprocess"])
        workflow.add_node("analyze", nodes["analyze"])
        workflow.add_node("validate", nodes["validate"])
        workflow.add_node("forecast", nodes["forecast"])
        workflow.add_node("report", nodes["report"])
        workflow.add_edge("preprocess", "analyze")
        workflow.add_edge("analyze", "validate")
        workflow.add_edge("validate", "forecast")
        workflow.add_edge("forecast", "report")
        workflow.add_edge("report", END)
        workflow.set_entry_point("preprocess")
        return workflow.compile()

    def run(self) -> dict:
        # 1. Load data and create slices
        print("Start loading data...")
        data_path = self.config.get('data_path')
        df = DataLoader.load_data(data_path)
        date_column = self.config.get('date_column', 'date')
        value_column = self.config.get('value_column', 'OT')
        df_ts = DataPreprocessor.convert_to_time_series(df, date_column, value_column)
        num_slices = self.config.get('num_slices', 10)
        input_length = self.config.get('input_length', 512)
        horizon = self.config.get('horizon', 96)
        slices = DataSplitter.create_slices(df_ts, num_slices, input_length, horizon)
        all_results = []
        
        # Add delay between slices to avoid rate limiting
        import time
        delay_between_slices = 3  # seconds
        
        print(f"Processing {len(slices)} slices with {delay_between_slices}s delay between slices...")
        print("=" * 60)
        
        # Collect all slice results
        for i, s in enumerate(slices):
            slice_start_time = time.time()
            print(f"Processing slice {i+1}/{len(slices)} (ID: {s['slice_id']})...")
            
            validation_data = s['validation']
            test_data = s['test']
            slice_info = {
                'slice_id': s['slice_id'],
                'validation_start': s['validation_start'],
                'validation_end': s['validation_end'],
                'test_start': s['test_start'],
                'test_end': s['test_end'],
            }
            # Build initial state for this slice
            state = {
                "validation_data": validation_data,
                "test_data": test_data,
                "slice_info": slice_info,
                "config": self.config
            }
            if self.debug:
                trace = []
                for chunk in self.graph.stream(state):
                    trace.append(chunk)
                final_state = trace[-1]
            else:
                final_state = self.graph.invoke(state)
            
            # Debug: Check final_state content
            print(f"Slice {s['slice_id']+1} final_state keys: {list(final_state.keys())}")
            print(f"Slice {s['slice_id']+1} forecast_result exists: {'forecast_result' in final_state}")
            if 'forecast_result' in final_state:
                forecast_result = final_state['forecast_result']
                print(f"Slice {s['slice_id']+1} forecast_result keys: {list(forecast_result.keys()) if forecast_result else 'None'}")
            
            all_results.append(final_state)
            
            # Calculate and display slice processing time
            slice_end_time = time.time()
            slice_duration = slice_end_time - slice_start_time
            print(f"Slice {i+1} completed in {slice_duration:.2f} seconds")
            
            # Add delay between slices (except for the last slice)
            if i < len(slices) - 1:
                print(f"Waiting {delay_between_slices} seconds before next slice...")
                time.sleep(delay_between_slices)
            
            print("-" * 40)
        
        print(f"All {len(slices)} slices processed successfully!")
        print("=" * 60)
        
        # Aggregate results from all slices
        aggregated_results = self._aggregate_slice_results(all_results)
        
        return {
            "all_results": all_results, 
            "aggregated_results": aggregated_results,
            "report": all_results[-1].get("report") if all_results else None
        }
    
    def _aggregate_slice_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all slices by averaging predictions and metrics"""
        print("Aggregating results from all slices...")
        
        if not all_results:
            print("Warning: No results to aggregate")
            return {}
        
        # Debug: Print structure of first result
        print(f"Number of results to aggregate: {len(all_results)}")
        if all_results:
            first_result = all_results[0]
            print(f"Keys in first result: {list(first_result.keys())}")
            if 'forecast_result' in first_result:
                forecast_result = first_result['forecast_result']
                print(f"Keys in forecast_result: {list(forecast_result.keys()) if forecast_result else 'None'}")
                if forecast_result:
                    print(f"individual_predictions keys: {list(forecast_result.get('individual_predictions', {}).keys())}")
                    print(f"ensemble_predictions: {forecast_result.get('ensemble_predictions', {})}")
                    print(f"test_metrics keys: {list(forecast_result.get('test_metrics', {}).keys())}")
                    # Debug: Print test_metrics content
                    test_metrics = forecast_result.get('test_metrics', {})
                    print(f"Test metrics content:")
                    for model, metrics in test_metrics.items():
                        print(f"  {model}: {metrics}")
            else:
                # Try to find forecast_result in report
                if 'report' in first_result:
                    print("Checking report content...")
                    report = first_result['report']
                    if isinstance(report, dict) and 'forecast_result' in report:
                        print(f"Found forecast_result in report: {list(report['forecast_result'].keys())}")
                        test_metrics = report['forecast_result'].get('test_metrics', {})
                        print(f"Test metrics from report:")
                        for model, metrics in test_metrics.items():
                            print(f"  {model}: {metrics}")
        
        # Collect all individual predictions and ensemble predictions
        all_individual_predictions = {}
        all_ensemble_predictions = []
        all_test_metrics = {}
        all_forecast_metrics = {}
        
        # Collect predictions from each slice
        for i, result in enumerate(all_results):
            print(f"Processing slice {i+1}/{len(all_results)}")
            
            # Try to get forecast_result from different possible locations
            forecast_result = None
            
            # First, try direct forecast_result
            if 'forecast_result' in result:
                forecast_result = result['forecast_result']
                print(f"Slice {i+1}: Found forecast_result directly")
            # If not found, try to extract from report
            elif 'report' in result and isinstance(result['report'], dict):
                if 'forecast_result' in result['report']:
                    forecast_result = result['report']['forecast_result']
                    print(f"Slice {i+1}: Found forecast_result in report")
                else:
                    print(f"Slice {i+1}: No forecast_result found in report")
            else:
                print(f"Slice {i+1}: No forecast_result found anywhere")
            
            if not forecast_result:
                print(f"Warning: No forecast_result for slice {i+1}")
                continue
            
            # Collect individual predictions
            individual_predictions = forecast_result.get('individual_predictions', {})
            print(f"Slice {i+1} individual_predictions: {len(individual_predictions)} models")
            for model_name, predictions in individual_predictions.items():
                if model_name not in all_individual_predictions:
                    all_individual_predictions[model_name] = []
                all_individual_predictions[model_name].append(predictions)
            
            # Collect ensemble predictions
            ensemble_predictions = forecast_result.get('ensemble_predictions', {})
            if ensemble_predictions and 'predictions' in ensemble_predictions:
                all_ensemble_predictions.append(ensemble_predictions['predictions'])
                print(f"Slice {i+1} ensemble_predictions: {len(ensemble_predictions['predictions'])} values")
            else:
                print(f"Warning: No ensemble_predictions for slice {i+1}")
            
            # Collect test metrics
            test_metrics = forecast_result.get('test_metrics', {})
            print(f"Slice {i+1} test_metrics: {len(test_metrics)} models")
            for model_name, metrics in test_metrics.items():
                if model_name not in all_test_metrics:
                    all_test_metrics[model_name] = {'mse': [], 'mae': [], 'mape': []}
                all_test_metrics[model_name]['mse'].append(metrics.get('mse', float('inf')))
                all_test_metrics[model_name]['mae'].append(metrics.get('mae', float('inf')))
                all_test_metrics[model_name]['mape'].append(metrics.get('mape', float('inf')))
            
            # Collect forecast metrics
            forecast_metrics = forecast_result.get('forecast_metrics', {})
            print(f"Slice {i+1} forecast_metrics: {len(forecast_metrics)} models")
            for model_name, metrics in forecast_metrics.items():
                if model_name not in all_forecast_metrics:
                    all_forecast_metrics[model_name] = {'mean': [], 'std': [], 'min': [], 'max': [], 'range': []}
                all_forecast_metrics[model_name]['mean'].append(metrics.get('mean', 0))
                all_forecast_metrics[model_name]['std'].append(metrics.get('std', 0))
                all_forecast_metrics[model_name]['min'].append(metrics.get('min', 0))
                all_forecast_metrics[model_name]['max'].append(metrics.get('max', 0))
                all_forecast_metrics[model_name]['range'].append(metrics.get('range', 0))
        
        print(f"Collected data summary:")
        print(f"  Individual predictions: {len(all_individual_predictions)} models")
        print(f"  Ensemble predictions: {len(all_ensemble_predictions)} slices")
        print(f"  Test metrics: {len(all_test_metrics)} models")
        print(f"  Forecast metrics: {len(all_forecast_metrics)} models")
        
        # Calculate averaged individual predictions
        averaged_individual_predictions = {}
        for model_name, predictions_list in all_individual_predictions.items():
            if predictions_list:
                # Convert to numpy arrays for easier averaging
                predictions_array = np.array(predictions_list)
                # Average across all slices
                averaged_predictions = np.mean(predictions_array, axis=0)
                averaged_individual_predictions[model_name] = averaged_predictions.tolist()
        
        # Calculate averaged ensemble predictions
        averaged_ensemble_predictions = {}
        if all_ensemble_predictions:
            ensemble_array = np.array(all_ensemble_predictions)
            averaged_ensemble = np.mean(ensemble_array, axis=0)
            averaged_ensemble_predictions = {
                'predictions': averaged_ensemble.tolist(),
                'method_used': 'average_across_slices',
                'num_slices': len(all_results)
            }
        
        # Calculate averaged test metrics
        averaged_test_metrics = {}
        for model_name, metrics_list in all_test_metrics.items():
            averaged_test_metrics[model_name] = {
                'mse': np.mean(metrics_list['mse']),
                'mae': np.mean(metrics_list['mae']),
                'mape': np.mean(metrics_list['mape'])
            }
        
        # Calculate averaged forecast metrics
        averaged_forecast_metrics = {}
        for model_name, metrics_list in all_forecast_metrics.items():
            averaged_forecast_metrics[model_name] = {
                'mean': np.mean(metrics_list['mean']),
                'std': np.mean(metrics_list['std']),
                'min': np.mean(metrics_list['min']),
                'max': np.mean(metrics_list['max']),
                'range': np.mean(metrics_list['range'])
            }
        
        # Create aggregated results
        aggregated_results = {
            'individual_predictions': averaged_individual_predictions,
            'ensemble_predictions': averaged_ensemble_predictions,
            'test_metrics': averaged_test_metrics,
            'forecast_metrics': averaged_forecast_metrics,
            'aggregation_info': {
                'num_slices': len(all_results),
                'aggregation_method': 'average',
                'slice_ids': [result.get('slice_info', {}).get('slice_id', i) for i, result in enumerate(all_results)]
            }
        }
        
        print(f"Aggregated results from {len(all_results)} slices")
        print(f"Final aggregated results summary:")
        print(f"  Individual predictions: {len(averaged_individual_predictions)} models")
        print(f"  Ensemble predictions: {'Yes' if averaged_ensemble_predictions else 'No'}")
        print(f"  Test metrics: {len(averaged_test_metrics)} models")
        print(f"  Forecast metrics: {len(averaged_forecast_metrics)} models")
        
        return aggregated_results 