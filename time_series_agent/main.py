#!/usr/bin/env python3
"""
Time Series Prediction Agent - Industrial-grade Entry Point
This script orchestrates the full time series agent workflow, similar to tradingagents' main entry.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from graph.agent_graph import TimeSeriesAgentGraph
from config.default_config import DEFAULT_CONFIG

if __name__ == "__main__":
    print("=" * 60)
    print("Time Series Prediction Agent - Industrial-grade System")
    print("=" * 60)

    # 1. Create and customize config
    config = DEFAULT_CONFIG.copy()
    # Example customizations (edit as needed):
    config["llm_provider"] = "openai"
    config["llm_model"] = "gpt-4o"
    config["num_slices"] = 25  # User's configuration
    config["input_length"] = 512
    config["horizon"] = 96  # User's configuration
    config["data_path"] = "../dataset/ETT-small/ETTh2.csv"  # User's configuration
    config["debug"] = True
    config["verbose"] = True
    # ... add more custom config as needed

    # 2. Check for API key in environment
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Please set the OPENAI_API_KEY environment variable before running.")
        sys.exit(1)

    # 3. Initialize orchestrator
    print("Initializing Time Series Agent Graph...")
    graph = TimeSeriesAgentGraph(config=config, model=config["llm_model"], debug=config["debug"])

    # 4. Run the workflow with enhanced monitoring and delays
    print("Running the time series agent workflow...")
    print(f"Configuration: {config['num_slices']} slices, {config['horizon']} horizon steps")
    print("=" * 60)
    
    # Add enhanced delay between slices to avoid rate limiting
    import time
    delay_between_slices = 5  # Increased delay to 5 seconds
    delay_between_agents = 2  # Delay between agent calls
    
    start_time = time.time()
    results = graph.run()
    end_time = time.time()
    
    # Calculate and display execution time
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    print(f"Average time per slice: {execution_time/config['num_slices']:.2f} seconds")
    
    # Add delay after completion
    print("Adding final delay to ensure API rate limit compliance...")
    time.sleep(delay_between_slices)

    # 5. Save results to file
    print("\n" + "=" * 60)
    print("Workflow execution completed!")
    print("=" * 60)
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete results (including all slice results)
    complete_report_filename = f"complete_time_series_report_{timestamp}.json"
    complete_report_path = results_dir / complete_report_filename
    
    try:
        with open(complete_report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Complete results saved to: {complete_report_path}")
    except Exception as e:
        print(f"Error saving complete results: {e}")
    
    # Save aggregated results separately (final averaged predictions)
    if results.get("aggregated_results"):
        aggregated_report_filename = f"aggregated_forecast_results_{timestamp}.json"
        aggregated_report_path = results_dir / aggregated_report_filename
        
        aggregated_summary = {
            "timestamp": timestamp,
            "aggregation_info": results["aggregated_results"]["aggregation_info"],
            "final_individual_predictions": results["aggregated_results"]["individual_predictions"],
            "final_ensemble_predictions": results["aggregated_results"]["ensemble_predictions"],
            "final_test_metrics": results["aggregated_results"]["test_metrics"],
            "final_forecast_metrics": results["aggregated_results"]["forecast_metrics"]
        }
        
        try:
            with open(aggregated_report_path, 'w', encoding='utf-8') as f:
                json.dump(aggregated_summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"Aggregated forecast results saved to: {aggregated_report_path}")
        except Exception as e:
            print(f"Error saving aggregated results: {e}")
    
    # Print summary of aggregated results
    if results.get("aggregated_results"):
        print("\n" + "=" * 60)
        print("FINAL AGGREGATED FORECAST RESULTS")
        print("=" * 60)
        
        agg_info = results["aggregated_results"]["aggregation_info"]
        print(f"Number of slices processed: {agg_info['num_slices']}")
        print(f"Aggregation method: {agg_info['aggregation_method']}")
        
        # Print final ensemble metrics
        if results["aggregated_results"]["test_metrics"].get("ensemble"):
            ensemble_metrics = results["aggregated_results"]["test_metrics"]["ensemble"]
            print(f"\nFinal Ensemble Performance:")
            print(f"  MSE: {ensemble_metrics['mse']:.4f}")
            print(f"  MAE: {ensemble_metrics['mae']:.4f}")
            print(f"  MAPE: {ensemble_metrics['mape']:.2f}%")
        
        # Print individual model metrics
        print(f"\nIndividual Model Performance (averaged across slices):")
        for model_name, metrics in results["aggregated_results"]["test_metrics"].items():
            if model_name != "ensemble":
                print(f"  {model_name}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")

    print("\nExperiment completed!") 