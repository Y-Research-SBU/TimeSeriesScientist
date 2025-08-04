"""
Report Agent for Time Series Prediction
Report Agent - responsible for generating comprehensive reports, summarizing results, and generating final outputs.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from datetime import datetime
import json

from agents.memory import ExperimentMemory
from utils.visualization_utils import TimeSeriesVisualizer, ReportVisualizer
from utils.file_utils import FileSaver, FileManager
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """
You are the Executive Reporting Agent for an industrial time series forecasting experiment.

Background:
- You are skilled in synthesizing complex experimental results into clear, actionable reports for both technical and business stakeholders.
- Your reports must be accurate, insightful, and tailored to drive future improvements.

Your responsibilities:
- Summarize the experiment's key findings, including data characteristics, model performance, and ensemble results.
- Highlight strengths, weaknesses, and any unexpected outcomes.
- Provide clear recommendations for future work, including potential improvements in data, modeling, or process.
- Structure your report for clarity, using markdown for formatting.
- Always ensure your report is concise, actionable, and suitable for both technical and non-technical audiences.

You have access to:
- The full experiment summary (as a Python dict)
- All relevant metrics, visualizations, and model details

Your output will be delivered to project stakeholders and used to guide future forecasting efforts.
"""

def convert_to_json_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def get_report_prompt(experiment_summary: dict) -> str:
    # Convert to JSON serializable format
    serializable_summary = convert_to_json_serializable(experiment_summary)
    
    return f"""
You are a time series forecasting report expert.

Given the following experiment summary:
{json.dumps(serializable_summary, indent=2)}

Please:
1. Write a concise executive summary.
2. Summarize key findings and model performance.
3. Highlight any issues or limitations.

IMPORTANT: Return your answer ONLY as a markdown string.
"""

class ReportAgent:
    """
    Report Agent
    Responsible for generating comprehensive reports, summarizing results, and generating final outputs.
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
        self.report_visualizer = ReportVisualizer(self.config)
        self.file_manager = FileManager(self.config.get('output_dir', 'results'))

    def run(self, experiment_summary: dict):
        """Run the report agent to generate comprehensive report"""
        import time
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        logger.info("Generating comprehensive report...")
        
        # Add retry mechanism for rate limiting
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                prompt = get_report_prompt(experiment_summary)
                response = self.llm.invoke([
                    SystemMessage(content=REPORT_SYSTEM_PROMPT), 
                    HumanMessage(content=prompt)
                ])
                
                if response and response.content:
                    logger.info("Report generated successfully")
                    return response.content
                else:
                    logger.warning("Empty response from LLM")
                    return self._generate_fallback_report(experiment_summary)
                    
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts")
                        return self._generate_fallback_report(experiment_summary)
                else:
                    logger.error(f"Error in report agent: {e}")
                    return self._generate_fallback_report(experiment_summary)
        
        return self._generate_fallback_report(experiment_summary)
    
    def _generate_fallback_report(self, experiment_summary: dict) -> str:
        """Generate fallback report when LLM fails"""
        logger.info("Generating fallback report...")
        
        try:
            # Extract key information from experiment summary
            forecast_result = experiment_summary.get('forecast_result', {})
            test_metrics = forecast_result.get('test_metrics', {})
            
            # Get ensemble metrics if available
            ensemble_metrics = test_metrics.get('ensemble', {})
            
            # Create basic report
            report = f"""
# Time Series Forecasting Experiment Report

## Executive Summary
This report summarizes the results of a time series forecasting experiment conducted on industrial data.

## Key Findings

### Model Performance
"""
            
            # Add model performance metrics
            for model_name, metrics in test_metrics.items():
                if model_name != 'ensemble':
                    mse = metrics.get('mse', 'N/A')
                    mae = metrics.get('mae', 'N/A')
                    mape = metrics.get('mape', 'N/A')
                    report += f"- **{model_name}**: MSE={mse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%\n"
            
            # Add ensemble performance
            if ensemble_metrics:
                ensemble_mse = ensemble_metrics.get('mse', 'N/A')
                ensemble_mae = ensemble_metrics.get('mae', 'N/A')
                ensemble_mape = ensemble_metrics.get('mape', 'N/A')
                report += f"\n### Ensemble Performance\n"
                report += f"- **Ensemble**: MSE={ensemble_mse:.4f}, MAE={ensemble_mae:.4f}, MAPE={ensemble_mape:.2f}%\n"
            
            report += f"""
## Data Characteristics
- **Dataset**: {experiment_summary.get('slice_info', {}).get('slice_id', 'Unknown')}
- **Forecast Horizon**: {experiment_summary.get('config', {}).get('horizon', 'Unknown')} steps

## Recommendations
1. Consider using ensemble methods for improved accuracy
2. Monitor model performance over time
3. Collect additional data if possible for better training

## Limitations
- This is a fallback report generated due to API rate limiting
- Limited analysis depth compared to full LLM-generated report
- Performance metrics may need further validation

*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating fallback report: {e}")
            return f"""
# Time Series Forecasting Experiment Report

## Executive Summary
A time series forecasting experiment was conducted successfully.

## Status
- **Status**: Completed with fallback reporting
- **Reason**: API rate limiting prevented full report generation
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Note
This is a minimal fallback report. Please check the detailed results in the output files for complete information.
""" 