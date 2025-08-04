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
    
    # def _analyze_basic_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
    #     """Basic statistical analysis"""
    #     logger.info("Analyzing basic statistics...")
        
    #     basic_stats = self.analyzer.get_basic_stats(data)
        
    #     # Add more statistical metrics
    #     for col in data.columns:
    #         series = data[col].dropna()
    #         if len(series) > 0:
    #             basic_stats[col].update({
    #                 'skewness': stats.skew(series),
    #                 'kurtosis': stats.kurtosis(series),
    #                 'coefficient_of_variation': series.std() / series.mean() if series.mean() != 0 else 0,
    #                 'range': series.max() - series.min(),
    #                 'iqr': series.quantile(0.75) - series.quantile(0.25)
    #             })
        
    #     return basic_stats
    
    # def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
    #     """Trend analysis"""
    #     logger.info("Analyzing trends...")
        
    #     trend_results = {}
        
    #     for col in data.columns:
    #         series = data[col].dropna()
    #         if len(series) < 2:
    #             continue
            
    #         # Linear trend analysis
    #         x = np.arange(len(series))
    #         slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
    #         # Trend strength
    #         trend_strength = abs(r_value)
            
    #         # Trend direction
    #         if slope > 0:
    #             trend_direction = 'increasing'
    #         elif slope < 0:
    #             trend_direction = 'decreasing'
    #         else:
    #             trend_direction = 'stable'
            
    #         # Trend significance
    #         trend_significant = p_value < 0.05
            
    #         trend_results[col] = {
    #             'has_trend': trend_significant and trend_strength > 0.3,
    #             'trend_direction': trend_direction,
    #             'slope': slope,
    #             'intercept': intercept,
    #             'r_squared': r_value ** 2,
    #             'p_value': p_value,
    #             'trend_strength': trend_strength,
    #             'trend_significant': trend_significant,
    #             'std_error': std_err
    #         }
        
    #     return trend_results
    
    # def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
    #     """Seasonality analysis"""
    #     logger.info("Analyzing seasonality...")
        
    #     seasonality_results = {}
        
    #     for col in data.columns:
    #         series = data[col].dropna()
    #         if len(series) < 4:
    #             continue
            
    #         # Detect seasonal periods
    #         seasonal_periods = self._detect_seasonal_periods(series)
            
    #         # Seasonal decomposition
    #         decomposition_results = self._perform_seasonal_decomposition(series, seasonal_periods)
            
    #         # Seasonal strength
    #         seasonal_strength = self._calculate_seasonal_strength(series, decomposition_results)
            
    #         seasonality_results[col] = {
    #             'has_seasonality': seasonal_strength > 0.1,
    #             'seasonal_periods': seasonal_periods,
    #             'seasonal_strength': seasonal_strength,
    #             'decomposition': decomposition_results
    #         }
        
    #     return seasonality_results
    
    # def _detect_seasonal_periods(self, series: pd.Series) -> List[int]:
    #     """Detect seasonal periods"""
    #     periods = []
        
    #     # Check common seasonal periods
    #     for period in self.seasonal_periods:
    #         if len(series) >= period * 2:
    #             # Calculate autocorrelation coefficient for that period
    #             autocorr = series.autocorr(lag=period)
    #             if abs(autocorr) > 0.3:  # Threshold can be adjusted
    #                 periods.append(period)
        
    #     # If no obvious seasonality detected, try to auto-detect
    #     if not periods and len(series) > 50:
    #         # Use FFT to detect periodicity
    #         fft = np.fft.fft(series)
    #         freqs = np.fft.fftfreq(len(series))
            
    #         # Find main frequency
    #         main_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
    #         main_freq = freqs[main_freq_idx]
            
    #         if main_freq != 0:
    #             period = int(1 / abs(main_freq))
    #             if 2 <= period <= len(series) // 2:
    #                 periods.append(period)
        
    #     return periods
    
    # def _perform_seasonal_decomposition(self, series: pd.Series, periods: List[int]) -> Dict[str, Any]:
    #     """Perform seasonal decomposition"""
    #     if not periods:
    #         return {}
        
    #     # Use the shortest period for decomposition
    #     period = min(periods)
        
    #     try:
    #         decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
            
    #         return {
    #             'trend': decomposition.trend.tolist(),
    #             'seasonal': decomposition.seasonal.tolist(),
    #             'residual': decomposition.resid.tolist(),
    #             'period': period
    #         }
    #     except Exception as e:
    #         logger.warning(f"Seasonal decomposition failed: {e}")
    #         return {}
    
    # def _calculate_seasonal_strength(self, series: pd.Series, decomposition: Dict[str, Any]) -> float:
    #     """Calculate seasonal strength"""
    #     if not decomposition or 'seasonal' not in decomposition:
    #         return 0.0
        
    #     seasonal = np.array(decomposition['seasonal'])
    #     residual = np.array(decomposition['residual'])
        
    #     # Seasonal strength = Seasonal variance / (Seasonal variance + Residual variance)
    #     seasonal_var = np.var(seasonal)
    #     residual_var = np.var(residual)
        
    #     if seasonal_var + residual_var == 0:
    #         return 0.0
        
    #     return seasonal_var / (seasonal_var + residual_var)
    
    # def _analyze_stationarity(self, data: pd.DataFrame) -> Dict[str, Any]:
    #     """Stationarity test"""
    #     logger.info("Analyzing stationarity...")
        
    #     stationarity_results = {}
        
    #     for col in data.columns:
    #         series = data[col].dropna()
    #         if len(series) < 10:
    #             continue
            
    #         # ADF test
    #         adf_result = self._perform_adf_test(series)
            
    #         # KPSS test
    #         kpss_result = self._perform_kpss_test(series)
            
    #         # Combined judgment
    #         is_stationary = adf_result['is_stationary'] and kpss_result['is_stationary']
            
    #         stationarity_results[col] = {
    #             'is_stationary': is_stationary,
    #             'adf_test': adf_result,
    #             'kpss_test': kpss_result,
    #             'confidence': self._calculate_stationarity_confidence(adf_result, kpss_result)
    #         }
        
    #     return stationarity_results
    
    # def _perform_adf_test(self, series: pd.Series) -> Dict[str, Any]:
    #     """Perform ADF test"""
    #     try:
    #         adf_stat, p_value, critical_values, _ = adfuller(series)
            
    #         return {
    #             'statistic': adf_stat,
    #             'p_value': p_value,
    #             'critical_values': critical_values,
    #             'is_stationary': p_value < 0.05
    #         }
    #     except Exception as e:
    #         logger.warning(f"ADF test failed: {e}")
    #         return {
    #             'statistic': None,
    #             'p_value': None,
    #             'critical_values': None,
    #             'is_stationary': False
    #         }
    
    # def _perform_kpss_test(self, series: pd.Series) -> Dict[str, Any]:
    #     """Perform KPSS test"""
    #     try:
    #         kpss_stat, p_value, critical_values, _ = kpss(series)
            
    #         return {
    #             'statistic': kpss_stat,
    #             'p_value': p_value,
    #             'critical_values': critical_values,
    #             'is_stationary': p_value > 0.05
    #         }
    #     except Exception as e:
    #         logger.warning(f"KPSS test failed: {e}")
    #         return {
    #             'statistic': None,
    #             'p_value': None,
    #             'critical_values': None,
    #             'is_stationary': False
    #         }
    
    # def _calculate_stationarity_confidence(self, adf_result: Dict, kpss_result: Dict) -> float:
    #     """Calculate stationarity confidence"""
    #     confidence = 0.0
        
    #     if adf_result['is_stationary']:
    #         confidence += 0.5
    #     if kpss_result['is_stationary']:
    #         confidence += 0.5
        
    #     return confidence
    
    # def _analyze_autocorrelation(self, data: pd.DataFrame) -> Dict[str, Any]:
    #     """Autocorrelation analysis"""
    #     logger.info("Analyzing autocorrelation...")
        
    #     autocorr_results = {}
        
    #     for col in data.columns:
    #         series = data[col].dropna()
    #         if len(series) < 10:
    #             continue
            
    #         # Calculate autocorrelation coefficients
    #         autocorr_values = []
    #         for lag in range(1, min(self.max_lag, len(series) // 2)):
    #             autocorr = series.autocorr(lag=lag)
    #             if not pd.isna(autocorr):
    #                 autocorr_values.append(autocorr)
    #             else:
    #                 break
            
    #         # Find significant autocorrelation lags
    #         significant_lags = []
    #         for i, autocorr in enumerate(autocorr_values):
    #             if abs(autocorr) > 0.2:  # Threshold can be adjusted
    #                 significant_lags.append(i + 1)
            
    #         autocorr_results[col] = {
    #             'autocorr_values': autocorr_values,
    #             'significant_lags': significant_lags,
    #             'max_autocorr': max(autocorr_values) if autocorr_values else 0,
    #             'min_autocorr': min(autocorr_values) if autocorr_values else 0,
    #             'autocorr_range': max(autocorr_values) - min(autocorr_values) if autocorr_values else 0
    #         }
        
    #     return autocorr_results
    
    # def _summarize_characteristics(self, basic_stats: Dict, trend_analysis: Dict, 
    #                              seasonality_analysis: Dict, stationarity_analysis: Dict,
    #                              autocorrelation_analysis: Dict) -> Dict[str, Any]:
    #     """Summarize data characteristics"""
    #     logger.info("Summarizing data characteristics...")
        
    #     characteristics = {}
        
    #     for col in basic_stats.keys():
    #         char = {
    #             'length': basic_stats[col].get('count', 0),
    #             'has_trend': trend_analysis.get(col, {}).get('has_trend', False),
    #             'trend_direction': trend_analysis.get(col, {}).get('trend_direction', 'unknown'),
    #             'has_seasonality': seasonality_analysis.get(col, {}).get('has_seasonality', False),
    #             'is_stationary': stationarity_analysis.get(col, {}).get('is_stationary', False),
    #             'seasonal_periods': seasonality_analysis.get(col, {}).get('seasonal_periods', []),
    #             'significant_autocorr_lags': autocorrelation_analysis.get(col, {}).get('significant_lags', []),
    #             'data_type': self._classify_data_type(
    #                 trend_analysis.get(col, {}),
    #                 seasonality_analysis.get(col, {}),
    #                 stationarity_analysis.get(col, {})
    #             )
    #         }
    #         characteristics[col] = char
        
    #     return characteristics
    
    # def _classify_data_type(self, trend_info: Dict, seasonality_info: Dict, stationarity_info: Dict) -> str:
    #     """Classify data type"""
    #     has_trend = trend_info.get('has_trend', False)
    #     has_seasonality = seasonality_info.get('has_seasonality', False)
    #     is_stationary = stationarity_info.get('is_stationary', False)
        
    #     if has_trend and has_seasonality:
    #         return 'trend_seasonal'
    #     elif has_trend:
    #         return 'trend'
    #     elif has_seasonality:
    #         return 'seasonal'
    #     elif is_stationary:
    #         return 'stationary'
    #     else:
    #         return 'random_walk'
    
    # def _generate_model_recommendations(self, characteristics: Dict[str, Any]) -> Dict[str, List[str]]:
    #     """Generate model selection suggestions"""
    #     logger.info("Generating model recommendations...")
        
    #     recommendations = {}
        
    #     for col, char in characteristics.items():
    #         recommended_models = []
            
    #         data_type = char.get('data_type', 'unknown')
            
    #         if data_type == 'trend_seasonal':
    #             recommended_models = ['Prophet', 'SARIMA', 'LSTM', 'ExponentialSmoothing']
    #         elif data_type == 'trend':
    #             recommended_models = ['LinearRegression', 'RandomForest', 'LSTM', 'ARIMA']
    #         elif data_type == 'seasonal':
    #             recommended_models = ['SARIMA', 'ExponentialSmoothing', 'Prophet', 'LSTM']
    #         elif data_type == 'stationary':
    #             recommended_models = ['ARMA', 'ARIMA', 'RandomForest', 'SVR']
    #         else:  # random_walk
    #             recommended_models = ['RandomWalk', 'ARIMA', 'LSTM', 'RandomForest']
            
    #         recommendations[col] = recommended_models
        
    #     return recommendations
    
    # def _generate_analysis_visualizations(self, data: pd.DataFrame, basic_stats: Dict,
    #                                     trend_analysis: Dict, seasonality_analysis: Dict,
    #                                     stationarity_analysis: Dict, autocorrelation_analysis: Dict,
    #                                     output_dir: str) -> Dict[str, str]:
    #     """Generate analysis visualizations"""
    #     logger.info("Generating analysis visualizations...")
        
    #     output_path = Path(output_dir)
    #     output_path.mkdir(parents=True, exist_ok=True)
        
    #     visualizations = {}
        
    #     try:
    #         # 1. Comprehensive analysis results plot
    #         analysis_plot_path = output_path / "analysis_results.png"
    #         visualizations['analysis_results'] = self.visualizer.plot_analysis_results(
    #             data, {
    #                 'basic_stats': basic_stats,
    #                 'trend_analysis': trend_analysis,
    #                 'seasonality_analysis': seasonality_analysis,
    #                 'stationarity_analysis': stationarity_analysis
    #             }, str(analysis_plot_path)
    #         )
            
    #         # 2. Autocorrelation plot
    #         autocorr_plot_path = output_path / "autocorrelation.png"
    #         visualizations['autocorrelation'] = self._plot_autocorrelation(
    #             data, autocorrelation_analysis, str(autocorr_plot_path)
    #         )
            
    #         # 3. Seasonal decomposition plot
    #         seasonal_plot_path = output_path / "seasonal_decomposition.png"
    #         visualizations['seasonal_decomposition'] = self._plot_seasonal_decomposition(
    #             data, seasonality_analysis, str(seasonal_plot_path)
    #         )
            
    #         logger.info(f"Generated {len(visualizations)} analysis visualizations")
            
    #     except Exception as e:
    #         logger.error(f"Analysis visualization generation failed: {e}")
    #         visualizations = {}
        
    #     return visualizations
    
    # def _plot_autocorrelation(self, data: pd.DataFrame, autocorr_analysis: Dict[str, Any], save_path: str) -> str:
    #     """Plot autocorrelation graph"""
    #     try:
    #         import matplotlib.pyplot as plt
            
    #         fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
    #         for i, col in enumerate(data.columns):
    #             if col in autocorr_analysis:
    #                 autocorr_values = autocorr_analysis[col]['autocorr_values']
    #                 lags = range(1, len(autocorr_values) + 1)
                    
    #                 axes[i].plot(lags, autocorr_values, 'o-', markersize=4)
    #                 axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    #                 axes[i].axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
    #                 axes[i].axhline(y=-0.2, color='r', linestyle='--', alpha=0.5)
    #                 axes[i].set_title(f'Autocorrelation - {col}')
    #                 axes[i].set_xlabel('Lag')
    #                 axes[i].set_ylabel('Autocorrelation')
    #                 axes[i].grid(True, alpha=0.3)
            
    #         plt.tight_layout()
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         plt.close()
            
    #         return save_path
            
    #     except Exception as e:
    #         logger.error(f"Autocorrelation plot failed: {e}")
    #         return ""
    
    # def _plot_seasonal_decomposition(self, data: pd.DataFrame, seasonality_analysis: Dict[str, Any], save_path: str) -> str:
    #     """Plot seasonal decomposition graph"""
    #     try:
    #         import matplotlib.pyplot as plt
            
    #         for col in data.columns:
    #             if col in seasonality_analysis and seasonality_analysis[col].get('has_seasonality'):
    #                 decomposition = seasonality_analysis[col].get('decomposition', {})
    #                 if decomposition:
    #                     fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                        
    #                     series = data[col].dropna()
    #                     axes[0].plot(series.index, series.values)
    #                     axes[0].set_title(f'Original Time Series - {col}')
    #                     axes[0].set_ylabel('Value')
                        
    #                     if 'trend' in decomposition:
    #                         axes[1].plot(series.index, decomposition['trend'])
    #                         axes[1].set_title('Trend')
    #                         axes[1].set_ylabel('Value')
                        
    #                     if 'seasonal' in decomposition:
    #                         axes[2].plot(series.index, decomposition['seasonal'])
    #                         axes[2].set_title('Seasonal')
    #                         axes[2].set_ylabel('Value')
                        
    #                     if 'residual' in decomposition:
    #                         axes[3].plot(series.index, decomposition['residual'])
    #                         axes[3].set_title('Residual')
    #                         axes[3].set_ylabel('Value')
                        
    #                     plt.tight_layout()
    #                     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #                     plt.close()
                        
    #                     return save_path
            
    #         return ""
            
    #     except Exception as e:
    #         logger.error(f"Seasonal decomposition plot failed: {e}")
    #         return ""
    
    # def _save_analysis_results(self, basic_stats: Dict, trend_analysis: Dict,
    #                          seasonality_analysis: Dict, stationarity_analysis: Dict,
    #                          autocorrelation_analysis: Dict, data_characteristics: Dict,
    #                          model_recommendations: Dict, output_dir: str):
    #     """Save analysis results"""
    #     logger.info("Saving analysis results...")
        
    #     output_path = Path(output_dir)
        
    #     # Save analysis report
    #     from utils.file_utils import FileSaver
    #     analysis_report = {
    #         'basic_stats': basic_stats,
    #         'trend_analysis': trend_analysis,
    #         'seasonality_analysis': seasonality_analysis,
    #         'stationarity_analysis': stationarity_analysis,
    #         'autocorrelation_analysis': autocorrelation_analysis,
    #         'data_characteristics': data_characteristics,
    #         'model_recommendations': model_recommendations
    #     }
        
    #     report_path = output_path / "analysis_report.json"
    #     FileSaver.save_json(analysis_report, report_path)
    #     logger.info(f"Analysis report saved to {report_path}")
    
    # def _update_memory(self, basic_stats: Dict, trend_analysis: Dict,
    #                   seasonality_analysis: Dict, stationarity_analysis: Dict,
    #                   autocorrelation_analysis: Dict, data_characteristics: Dict,
    #                   model_recommendations: Dict, visualizations: Dict[str, str]):
    #     """Update memory"""
    #     self.memory.store('basic_stats', basic_stats, 'analysis')
    #     self.memory.store('trend_analysis', trend_analysis, 'analysis')
    #     self.memory.store('seasonality_analysis', seasonality_analysis, 'analysis')
    #     self.memory.store('stationarity_analysis', stationarity_analysis, 'analysis')
    #     self.memory.store('autocorrelation_analysis', autocorrelation_analysis, 'analysis')
    #     self.memory.store('data_characteristics', data_characteristics, 'analysis')
    #     self.memory.store('model_recommendations', model_recommendations, 'models')
    #     self.memory.store('analysis_visualizations', visualizations, 'visualizations')
        
    #     # Record analysis history
    #     self.memory.add_history(
    #         'analysis',
    #         {
    #             'data_characteristics': data_characteristics,
    #             'model_recommendations': model_recommendations,
    #             'visualization_count': len(visualizations)
    #         }
    #     )
    
    # def get_analysis_summary(self) -> Dict[str, Any]:
    #     """Get analysis summary"""
    #     data_characteristics = self.memory.retrieve('data_characteristics', 'analysis')
    #     model_recommendations = self.memory.retrieve('model_recommendations', 'models')
        
    #     if not data_characteristics:
    #         return {}
        
    #     summary = {}
    #     for col, char in data_characteristics.items():
    #         summary[col] = {
    #             'data_type': char.get('data_type', 'unknown'),
    #             'has_trend': char.get('has_trend', False),
    #             'has_seasonality': char.get('has_seasonality', False),
    #             'is_stationary': char.get('is_stationary', False),
    #             'recommended_models': model_recommendations.get(col, [])[:3]  # Top 3 recommended models
    #         }
        
    #     return summary 