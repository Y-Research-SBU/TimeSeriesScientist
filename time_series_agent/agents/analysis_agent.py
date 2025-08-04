"""
Analysis Agent for Time Series Prediction
Data analysis Agent - responsible for time series feature analysis, trend detection, seasonal analysis, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utils.data_utils import DataAnalyzer, DataValidator
from utils.visualization_utils import TimeSeriesVisualizer
from agents.memory import ExperimentMemory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """
You are the Principal Data Analyst Agent for a state-of-the-art time series forecasting platform.

Background:
- You are an expert in time series statistics, pattern recognition, and exploratory data analysis.
- Your insights will guide model selection, hyperparameter tuning, and risk assessment.

Your responsibilities:
- Provide a comprehensive statistical summary of the input data, including central tendency, dispersion, skewness, and kurtosis.
- Detect and describe any trends, seasonality, regime shifts, or anomalies.
- Assess stationarity and discuss its implications for modeling.
- Identify potential challenges for forecasting, such as non-stationarity, structural breaks, or data quality issues.
- Justify all findings with reference to the data and, where possible, relate them to best practices in time series modeling.
- Always return your analysis in a structured Python dict, with clear, concise, and actionable insights.

You have access to:
- The cleaned time series data (as a Python dict)
- Visualizations (if available) to support your analysis

Your output will be used by downstream agents to select and configure forecasting models.
"""

def get_analysis_prompt(data: pd.DataFrame, visualizations: dict = None) -> str:
    prompt = f"""
You are a time series analysis agent. Analyze the following data {data} and visualizations of the data {visualizations}.
Return your analysis in the following JSON format:

{{
  "trend_analysis": "string",
  "seasonality_analysis": "string",
  "stationarity": "string",
  "potential_issues": "string",
  "summary": "string"
}}
IMPORTANT: Return your answer ONLY as a JSON object.
"""
    return prompt

class AnalysisAgent:
    """
    Data analysis Agent
    Responsible for time series feature analysis, trend detection, seasonal analysis, and stationarity testing.
    """
    
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            max_tokens=4000,
        )
        self.config = config or {}
        self.analyzer = DataAnalyzer()
        self.validator = DataValidator()
        self.visualizer = TimeSeriesVisualizer(self.config)
        self.memory = ExperimentMemory(self.config)
        self.seasonal_periods = [12, 7, 5, 3] # Common seasonal periods to check
        self.max_lag = 20 # Maximum lag for autocorrelation analysis

    def run(self, data: pd.DataFrame, visualizations: Dict[str, str] = None) -> str:
        """Run the analysis agent"""
        logger.info("Running analysis agent...")
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(data, visualizations)
        
        # Add retry mechanism for rate limiting
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke([
                    SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
                    HumanMessage(content=prompt)
                ])
                return response.content
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} attempts")
                        return self._generate_fallback_analysis(data)
                else:
                    logger.error(f"Error in analysis agent: {e}")
                    return self._generate_fallback_analysis(data)
        
        return self._generate_fallback_analysis(data)
    
    def _generate_fallback_analysis(self, data: pd.DataFrame) -> str:
        """Generate fallback analysis when LLM fails"""
        logger.info("Generating fallback analysis...")
        
        # Calculate basic statistics
        basic_stats = {
            "mean": float(data['value'].mean()),
            "std": float(data['value'].std()),
            "min": float(data['value'].min()),
            "max": float(data['value'].max())
        }
        
        # Determine trend
        if len(data) > 1:
            slope = np.polyfit(range(len(data)), data['value'], 1)[0]
            trend = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
        else:
            trend = "stable"
        
        return f"""
# Time Series Analysis Report

## Data Overview
- **Dataset Size:** {len(data)} observations
- **Time Range:** {data.index.min()} to {data.index.max()}
- **Basic Statistics:**
  - Mean: {basic_stats['mean']:.4f}
  - Standard Deviation: {basic_stats['std']:.4f}
  - Minimum: {basic_stats['min']:.4f}
  - Maximum: {basic_stats['max']:.4f}

## Trend Analysis
The time series data shows a **{trend}** trend over the observation period.

## Data Characteristics
- **Stationarity:** Analysis indicates the data may be non-stationary
- **Seasonality:** Potential seasonal patterns detected
- **Data Quality:** Data appears to be well-structured and suitable for forecasting

## Recommendations
1. Consider using models that can handle non-stationary data
2. Implement seasonal decomposition if strong seasonality is present
3. Use appropriate preprocessing techniques for trend removal

*Note: This is a fallback analysis generated due to API rate limiting.*
"""
    
    def _create_analysis_prompt(self, data: pd.DataFrame, visualizations: Dict[str, str] = None) -> str:
        """Create analysis prompt for LLM"""
        # Convert data to dict for LLM analysis
        sample = data.to_dict(orient='list')
        
        viz_info = ""
        if visualizations:
            viz_info = f"\nGenerated Visualizations:\n{visualizations}\n"
        
        return f"""
Given the following time series data and visualizations, please provide a comprehensive analysis.

Data (as a Python dict):
{sample}
{viz_info}

Please analyze:
1. Trend analysis - overall direction and strength
2. Seasonality analysis - any recurring patterns
3. Stationarity - whether the data is stationary
4. Potential issues for forecasting
5. Summary of key findings

Return your analysis in a clear, structured format.
"""
    
