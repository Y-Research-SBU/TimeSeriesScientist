"""
Preprocess Agent for Time Series Prediction
Data preprocessing Agent - responsible for data loading, cleaning, and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from utils.data_utils import DataLoader, DataPreprocessor, DataValidator, DataAnalyzer
from utils.visualization_utils import TimeSeriesVisualizer
from agents.memory import ExperimentMemory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREPROCESS_SYSTEM_PROMPT = """
You are the Data Preprocessing Chief Agent for an advanced time series forecasting system. 
Your mission is to ensure that all input data is of the highest possible quality before it enters the modeling pipeline.

Background:
- You have deep expertise in time series data cleaning, anomaly detection, and preparation for machine learning and statistical forecasting.
- You understand the downstream impact of preprocessing choices on model performance and interpretability.

Your responsibilities:
- Rigorously assess the quality of the input time series, identifying missing values, outliers, and structural issues.
- For each issue, recommend the most appropriate handling strategy, considering both statistical best practices and the needs of advanced forecasting models.
- Justify your recommendations with clear reasoning, referencing both the data characteristics and potential modeling implications.
- If relevant, suggest additional preprocessing steps (e.g., resampling, detrending, feature engineering) that could improve results.
- Always return your decisions in a structured Python dict, and ensure your reasoning is transparent and actionable.

You have access to:
- The raw time series data (as a Python dict)
- Any prior preprocessing history or known data issues

Your output will directly determine how the data is prepared for all subsequent analysis and modeling.
"""

class PreprocessLLMTools:
    def __init__(self, llm):
        self.llm = llm

    def get_preprocess_decision_prompt(self, data: pd.DataFrame) -> str:
        return f"""
You are a time series data preprocessing expert.

Given the following time series data (as a Python dict):

{data.to_dict(orient='list')}

Please:
1. Assess the overall data quality.
2. Recommend a missing value handling strategy (choose from: interpolate, forward_fill, backward_fill, mean, median, drop, zero).
3. Recommend an outlier handling strategy (choose from: clip, drop, zero, interpolate, ffill, bfill, mean, median, smooth).
4. Optionally, suggest any other preprocessing steps if needed.

Return your answer as a Python dict:
{
  "quality_assessment": "string",
  "missing_value_strategy": "string",
  "outlier_strategy": "string",
  "other_suggestions": "string"
}
"""

    def analyze_data_quality(self, data: pd.DataFrame) -> dict:
        prompt = self.get_preprocess_decision_prompt(data)
        response = self.llm.invoke([SystemMessage(content=PREPROCESS_SYSTEM_PROMPT), 
                                    HumanMessage(content=prompt)])
        return response.content

class PreprocessAgent:
    """
    Data preprocessing Agent
    Responsible for data loading, cleaning, validation, and visualization.
    """
    
    def __init__(self, model: str = "gpt-4o", config: dict = None):
        """
        Initialize the preprocessing agent.
        
        Args:
            config: Configuration dictionary
            memory: Experiment memory manager
        """
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            # max_tokens=4000,
        )
        self.tools = PreprocessLLMTools(self.llm)
        self.config = config
        self.visualizer = TimeSeriesVisualizer(self.config)
        
        # Get preprocessing configuration
        self.preprocess_config = config.get('preprocess')
        # self.outlier_method = self.preprocess_config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.preprocess_config.get('outlier_threshold', 1.5)
        
        logger.info("PreprocessAgent initialized")
        from agents.memory import ExperimentMemory
        self.memory = ExperimentMemory(self.config)
    
    def process(self, data: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Execute the complete preprocessing workflow.
        
        Args:
            data: Original data
            output_dir: Output directory
            
        Returns:
            Preprocessing result dictionary
        """
        logger.info("Starting data preprocessing...")
        
        # Debug: Print data index information
        logger.info(f"Input data index range: {data.index.min()} to {data.index.max()}")
        
        try:
            # 1. Data validation
            validation_result = self._validate_data(data)
            
            # 2. Initial data quality analysis for preprocessing strategies
            # Analyze data quality and get LLM recommendations for preprocessing
            initial_quality_analysis = self._analyze_data_quality(data, {})
            
            # Extract LLM-recommended strategies
            missing_value_strategy = initial_quality_analysis.get('recommended_strategies', {}).get('missing_value_strategy', 'interpolate')
            outlier_handle_strategy = initial_quality_analysis.get('recommended_strategies', {}).get('outlier_handle_strategy', 'clip')
            outlier_detect_strategy = initial_quality_analysis.get('recommended_strategies', {}).get('outlier_detect_strategy', 'iqr')
            
            logger.info(f"LLM recommended missing value strategy: {missing_value_strategy}")
            logger.info(f"LLM recommended outlier detect strategy: {outlier_detect_strategy}")
            logger.info(f"LLM recommended outlier handle strategy: {outlier_handle_strategy}")
            
            # 3. Clean data using LLM-recommended strategy
            cleaned_data = self._clean_data(data, missing_value_strategy)
            
            # 4. Detect outliers
            outlier_info = self._detect_outliers(cleaned_data, outlier_detect_strategy)
            
            # 5. Handle outliers using LLM-recommended strategy
            if outlier_info:
                cleaned_data = self._handle_outliers(cleaned_data, outlier_info, outlier_handle_strategy)
            
            # 6. Generate visualizations first
            visualizations = self._generate_visualizations(cleaned_data, output_dir)
            
            # 7. Generate comprehensive analysis report based on data and visualizations
            analysis_report = self._generate_comprehensive_analysis_report(cleaned_data, visualizations)
            
            # 8. Save preprocessing results
            self._save_preprocess_results(cleaned_data, analysis_report, output_dir)
            
            # 9. Update memory
            self._update_memory(cleaned_data, analysis_report, visualizations)
            
            result = {
                'cleaned_data': cleaned_data,
                'analysis_report': analysis_report,
                'outlier_info': outlier_info,
                'validation_result': validation_result,
                'visualizations': visualizations,
                'preprocess_config': {
                    'missing_strategy': missing_value_strategy,
                    'outlier_detect_strategy': outlier_detect_strategy,
                    'outlier_handle_strategy': outlier_handle_strategy,
                }
            }
            
            logger.info("Data preprocessing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            raise
    
    def run(self, data: pd.DataFrame, output_dir: str = None):
        """Run the preprocessing agent"""
        import time
        
        # Add small delay to avoid rate limiting
        time.sleep(0.5)
        
        return self.process(data, output_dir)
    
    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data"""
        logger.info("Validating data...")
        
        validator = DataValidator()
        validation_result = validator.validate_time_series(data)
        
        if not validation_result['is_valid']:
            logger.warning(f"Data validation issues found: {validation_result['errors']}")
        else:
            logger.info("Data validation passed")
        
        return validation_result
    
    def _clean_data(self, data: pd.DataFrame, missing_strategy: str) -> pd.DataFrame:
        """Clean data"""
        logger.info(f"Cleaning data with strategy: {missing_strategy}")
        
        preprocessor = DataPreprocessor()
        
        # Handle missing values
        cleaned_data = preprocessor.handle_missing_values(data, strategy=missing_strategy)
        
        # Check cleaning results
        missing_count = cleaned_data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Still have {missing_count} missing values after cleaning")
        else:
            logger.info("All missing values handled successfully")
        
        return cleaned_data
    
    def _detect_outliers(self, data: pd.DataFrame, outlier_strategy: str) -> Dict[str, Any]:
        """Detect outliers in the data"""
        logger.info("Detecting outliers...")
        
        preprocessor = DataPreprocessor()
        outlier_info = preprocessor.detect_outliers(
            data, 
            method=outlier_strategy,
            threshold=self.outlier_threshold
        )
        
        if outlier_info and any(outlier_info.values()):
            logger.info(f"Found outliers: {outlier_info}")
            return outlier_info
        else:
            logger.info("No outliers detected")
            return {}
    
    def _handle_outliers(self, data: pd.DataFrame, outlier_info: Dict[str, Any], outlier_strategy: str) -> pd.DataFrame:
        """Handle outliers based on detected outliers"""
        logger.info(f"Handling outliers with strategy: {outlier_strategy}")
        
        preprocessor = DataPreprocessor()
        
        # Handle outliers
        cleaned_data = preprocessor.handle_outliers(
            data, 
            outlier_info, 
            strategy=outlier_strategy
        )
        
        # Update data
        for col in data.columns:
            if col in outlier_info and outlier_info[col]:
                data[col] = cleaned_data[col]
        
        return cleaned_data
    
    def _generate_comprehensive_analysis_report(self, data: pd.DataFrame, visualizations: Dict[str, str]) -> Dict[str, Any]:
        """Analyze data quality and recommend preprocessing strategies using LLM"""
        logger.info("Analyzing data quality with LLM recommendations...")
        
        # Convert data to dict for LLM analysis
        sample = data.to_dict(orient='list')
        
        # Create comprehensive prompt for data analysis
        prompt = f"""
Given the following preprocessed time series data and generated visualizations, please provide a comprehensive analysis report.

Data (as a Python dict):
{sample}

Generated Visualizations:
{visualizations}

Note: This data has already been preprocessed - missing values and outliers have been handled.

Please provide a comprehensive analysis including:

1. Data Overview:
   - basic_stats: mean, std, min, max, trend
   - data_characteristics: seasonality, stationarity, patterns

2. Data Quality Assessment:
   - data_quality_score: overall quality score (0-1) after preprocessing
   - data_characteristics: key characteristics of the cleaned data

3. Insights from Visualizations:
   - key_patterns: patterns observed in the data
   - seasonal_components: any seasonal patterns
   - trend_analysis: overall trend direction and strength
   - distribution_characteristics: data distribution insights

4. Forecasting Readiness:
   - data_suitability: how suitable this data is for forecasting
   - potential_challenges: any challenges for forecasting models
   - data_strengths: strengths of this dataset

5. Model and Feature Recommendations:
   - model_suggestions: suitable model types for this data
   - feature_engineering: suggested features to create
   - preprocessing_effectiveness: how well the preprocessing worked

IMPORTANT: Return ONLY the JSON object below, with NO markdown formatting, NO code blocks, NO explanations. Just the raw JSON.

{{
    "data_overview": {{
        "basic_stats": {{
            "mean": float,
            "std": float,
            "min": float,
            "max": float,
            "trend": "string"
        }},
        "data_characteristics": {{
            "seasonality": "string",
            "stationarity": "string",
            "patterns": ["string"]
        }}
    }},
    "quality_assessment": {{
        "data_quality_score": float,
        "data_characteristics": "string"
    }},
    "visualization_insights": {{
        "key_patterns": ["string"],
        "seasonal_components": "string",
        "trend_analysis": "string",
        "distribution_characteristics": "string"
    }},
    "forecasting_readiness": {{
        "data_suitability": "string",
        "potential_challenges": ["string"],
        "data_strengths": ["string"]
    }},
    "recommendations": {{
        "model_suggestions": ["string"],
        "feature_engineering": ["string"],
        "preprocessing_effectiveness": "string"
    }}
}}
"""

        try:
            response = self.llm.invoke([
                # SystemMessage(content=PREPROCESS_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            # Check if response content is empty
            if not response.content or response.content.strip() == "":
                logger.warning("LLM returned empty response, using fallback analysis report")
                return self._generate_fallback_analysis_report(data)
            
            # Try to parse JSON response
            try:
                return json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response content: {response.content}")
                
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # Try to extract JSON using regex
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
                
                # If all parsing attempts fail, use fallback
                logger.warning("All JSON parsing attempts failed, using fallback analysis report")
                return self._generate_fallback_analysis_report(data)
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._generate_fallback_analysis_report(data)
    
    def _generate_fallback_analysis_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate fallback analysis report when LLM fails"""
        logger.info("Generating fallback analysis report...")
        
        # Calculate basic statistics
        basic_stats = {
            "mean": float(data['value'].mean()),
            "std": float(data['value'].std()),
            "min": float(data['value'].min()),
            "max": float(data['value'].max()),
            "trend": "stable"  # Default assumption
        }
        
        # Determine trend
        if len(data) > 1:
            slope = np.polyfit(range(len(data)), data['value'], 1)[0]
            if slope > 0.01:
                basic_stats["trend"] = "increasing"
            elif slope < -0.01:
                basic_stats["trend"] = "decreasing"
        
        return {
            "data_overview": {
                "basic_stats": basic_stats,
                "data_characteristics": {
                    "seasonality": "unknown",
                    "stationarity": "unknown",
                    "patterns": ["data_loaded_successfully"]
                }
            },
            "quality_assessment": {
                "data_quality_score": 0.8,  # Default score
                "data_characteristics": "Data has been preprocessed and is ready for analysis"
            },
            "visualization_insights": {
                "key_patterns": ["data_available_for_analysis"],
                "seasonal_components": "unknown",
                "trend_analysis": f"Overall trend appears to be {basic_stats['trend']}",
                "distribution_characteristics": "Data distribution available for analysis"
            },
            "forecasting_readiness": {
                "data_suitability": "suitable",
                "potential_challenges": ["limited_insights_due_to_llm_failure"],
                "data_strengths": ["preprocessed_data", "basic_statistics_available"]
            },
            "recommendations": {
                "model_suggestions": ["ARIMA", "ExponentialSmoothing", "LinearRegression"],
                "feature_engineering": ["lag_features", "rolling_statistics"],
                "preprocessing_effectiveness": "preprocessing_completed_successfully"
            }
        }
    
    def _analyze_data_quality(self, data: pd.DataFrame, outlier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data quality and recommend preprocessing strategies using LLM (for initial preprocessing)"""
        logger.info("Analyzing data quality for preprocessing strategies...")
        
        # Convert data to dict for LLM analysis
        sample = data.to_dict(orient='list')
        
        # Create prompt for initial preprocessing strategy recommendations
        prompt = f"""
Given the following time series data (as a Python dict):

{sample}

Please analyze the data quality and provide the following information as a JSON file:

1. Basic statistics for each column:
   - mean: float
   - std: float  
   - min: float
   - max: float
   - trend: 'increasing'/'decreasing'/'stable'

2. Missing value information:
   - missing_count: int (total missing values)
   - missing_percentage: float (percentage of missing values)

3. Outlier information:
   - outlier_count: int (total outliers detected)
   - outlier_percentage: float (percentage of outliers in the data, between 0 and 1)   

4. Data quality assessment:
   - data_quality_score: float (0-1, where 1 is perfect quality)
   - main_issues: list of strings (e.g., ['missing_values', 'outliers', 'noise', ...])

5. Recommended preprocessing strategies:
   - missing_value_strategy: string (choose from: 'interpolate', 'forward_fill', 'backward_fill', 'mean', 'median', 'drop', 'zero')
   - outlier_detect_strategy: string (choose from: 'iqr', 'zscore', 'percentile', 'none')
   - outlier_handle_strategy: string (choose from: 'clip', 'drop', 'interpolate', 'ffill', 'bfill', 'mean', 'median', 'smooth')

IMPORTANT:Return ONLY the JSON object below, with NO markdown formatting, NO code blocks, NO explanations. Just the raw JSON:

{{
    "basic_stats": {{
        "mean": float,
        "std": float,
        "min": float,
        "max": float,
        "trend": "string"
    }},
    "missing_info": {{
        "missing_count": int,
        "missing_percentage": float
    }},
    "outlier_info": {{
        "outlier_count": int,
        "outlier_percentage": float
    }},
    "quality_assessment": {{
        "data_quality_score": float,
        "main_issues": ["string"]
    }},
    "recommended_strategies": {{
        "missing_value_strategy": "string",
        "outlier_detect_strategy": "string",
        "outlier_handle_strategy": "string"
    }}
}}
"""
        
        try:
            response = self.llm.invoke([
                # SystemMessage(content=PREPROCESS_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            # Check if response content is empty
            if not response.content or response.content.strip() == "":
                logger.warning("LLM returned empty response, using fallback data quality analysis")
                return self._generate_fallback_data_quality_analysis(data, outlier_info)
            
            # Try to parse JSON response
            try:
                return json.loads(response.content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response content: {response.content}")
                
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass
                
                # Try to extract JSON using regex
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        pass
                
                # If all parsing attempts fail, use fallback
                logger.warning("All JSON parsing attempts failed, using fallback data quality analysis")
                return self._generate_fallback_data_quality_analysis(data, outlier_info)
                
        except Exception as e:
            logger.error(f"Error in LLM data quality analysis: {e}")
            return self._generate_fallback_data_quality_analysis(data, outlier_info)
    
    def _generate_fallback_data_quality_analysis(self, data: pd.DataFrame, outlier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback data quality analysis when LLM fails"""
        logger.info("Generating fallback data quality analysis...")
        
        # Calculate basic statistics
        basic_stats = {
            "mean": float(data['value'].mean()),
            "std": float(data['value'].std()),
            "min": float(data['value'].min()),
            "max": float(data['value'].max()),
            "trend": "stable"
        }
        
        # Determine trend
        if len(data) > 1:
            slope = np.polyfit(range(len(data)), data['value'], 1)[0]
            if slope > 0.01:
                basic_stats["trend"] = "increasing"
            elif slope < -0.01:
                basic_stats["trend"] = "decreasing"
        
        # Calculate missing values
        missing_count = data['value'].isnull().sum()
        missing_percentage = (missing_count / len(data)) * 100
        
        # Get outlier information from outlier_info
        outlier_count = outlier_info.get('outlier_count', 0)
        outlier_percentage = outlier_info.get('outlier_percentage', 0.0)
        
        # Determine data quality score
        quality_score = 1.0
        if missing_percentage > 0:
            quality_score -= missing_percentage / 100 * 0.3
        if outlier_percentage > 0.1:  # More than 10% outliers
            quality_score -= outlier_percentage * 0.2
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Determine main issues
        main_issues = []
        if missing_percentage > 0:
            main_issues.append("missing_values")
        if outlier_percentage > 0.05:
            main_issues.append("outliers")
        if len(main_issues) == 0:
            main_issues.append("none")
        
        return {
            "basic_stats": basic_stats,
            "missing_info": {
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_percentage)
            },
            "outlier_info": {
                "outlier_count": int(outlier_count),
                "outlier_percentage": float(outlier_percentage)
            },
            "quality_assessment": {
                "data_quality_score": float(quality_score),
                "main_issues": main_issues
            },
            "recommended_strategies": {
                "missing_value_strategy": "interpolate" if missing_percentage > 0 else "none",
                "outlier_detect_strategy": "iqr",
                "outlier_handle_strategy": "clip" if outlier_percentage > 0 else "none"
            }
        }
    
    def _calculate_quality_score(self, completeness: Dict, consistency: Dict, outlier_stats: Dict) -> float:
        """Calculate data quality score"""
        score = 100.0
        
        # Missing value penalty
        missing_percentages = completeness['missing_percentage'].values()
        avg_missing = np.mean(list(missing_percentages))
        score -= avg_missing * 2  # Each missing value penalizes 2 points
        
        # Duplicate value penalty
        duplicate_ratio = consistency['duplicate_rows'] / completeness['total_rows']
        score -= duplicate_ratio * 100
        
        # Outlier penalty
        if outlier_stats:
            outlier_percentages = [stats['percentage'] for stats in outlier_stats.values()]
            avg_outlier = np.mean(outlier_percentages)
            score -= avg_outlier * 0.5  # Each outlier penalizes 0.5 points
        
        return max(0, score)
    
    def _generate_visualizations(self, data: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """Generate visualizations using LLM prompts to decide what and how to plot"""
        logger.info("Generating visualizations using LLM decisions...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualizations = {}
        
        try:
            # Prompt LLM to decide what visualizations to create
            viz_decision_prompt = f"""
Given the following time series data:

Data shape: {data.shape}
Data columns: {list(data.columns)}

Please decide what visualizations would be most useful for understanding this data. 
Consider the data characteristics and quality issues.

Choose from these visualization types:
- time_series: Basic time series plot
- distribution: Histogram, box plot, KDE
- rolling_stats: Rolling mean, std, etc.
- autocorrelation: ACF/PACF plots
- seasonal_decomposition: Trend, seasonal, residual components

IMPORTANT: Return ONLY the JSON object below, with NO markdown formatting, NO code blocks, NO explanations. Just the raw JSON:
{{
    "visualizations": [
        {{
            "name": "string",
            "type": "string",
            "description": "string",
            "features": ["string"],
            "title": "string",
            "xlabel": "string", 
            "ylabel": "string",
            "additional_elements": ["string"],
            "plot_specific_params": {{}}
        }}
    ]
}}
"""
            
            response = self.llm.invoke([SystemMessage(content=PREPROCESS_SYSTEM_PROMPT),
                                        HumanMessage(content=viz_decision_prompt)])
            # print(response.content)
            # Extract visualization plan from LLM response
            viz_plan = json.loads(response.content)
            
            # Generate each visualization based on LLM decisions
            for viz_config in viz_plan.get("visualizations", []):
                viz_name = viz_config.get("name", "unknown")
                viz_type = viz_config.get("type", "time_series")
                
                if viz_type == "time_series":
                    plot_path = self._create_time_series_plot(data, viz_config, output_path)
                elif viz_type == "distribution":
                    plot_path = self._create_distribution_plot(data, viz_config, output_path)
                elif viz_type == "rolling_stats":
                    plot_path = self._create_rolling_stats_plot(data, viz_config, output_path)
                elif viz_type == "autocorrelation":
                    plot_path = self._create_autocorrelation_plot(data, viz_config, output_path)
                elif viz_type == "seasonal_decomposition":
                    plot_path = self._create_seasonal_decomposition_plot(data, viz_config, output_path)
                else:
                    logger.warning(f"Unknown visualization type: {viz_type}")
                    continue
                
                if plot_path:
                    visualizations[viz_name] = str(plot_path)
                    logger.info(f"Generated visualization '{viz_name}' at {plot_path}")
            
            logger.info(f"Generated {len(visualizations)} visualizations based on LLM decisions")
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            visualizations = {}
        
        return visualizations
    
    def _create_time_series_plot(self, data: pd.DataFrame, config: Dict[str, Any], output_path: Path) -> str:
        """Create time series plot based on LLM configuration"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6), facecolor='white')
            
            for col in data.columns:
                plt.plot(data.index, data[col], label=col, linewidth=2,color='#750014')
            
            # plt.title(config.get('title', 'Time Series Plot'))
            plt.xlabel(config.get('xlabel', 'Time'))
            plt.ylabel(config.get('ylabel', 'Value'))
            plt.legend()
            # plt.grid(True)

            ax = plt.gca()
            ax.set_facecolor("white")
            ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="black", alpha=0.3)  # 添加格子
            ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="black", alpha=0.2)   # 次级格子

            # 增大字体
            plt.title(config.get('title', 'Time Series Plot'), fontsize=16, fontweight='bold')
            plt.xlabel(config.get('xlabel', 'Time'), fontsize=14, fontweight='bold')
            plt.ylabel(config.get('ylabel', 'Value'), fontsize=14, fontweight='bold')
            
            # 增大刻度标签字体
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

            
            # Add additional elements as specified by LLM
            additional_elements = config.get('additional_elements', [])
            if 'rotate_x_labels' in additional_elements:
                plt.xticks(rotation=45)
            
            save_path = output_path / f"{config.get('name', 'time_series_plot')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Time series plot failed: {e}")
            return ""
    
    def _create_distribution_plot(self, data: pd.DataFrame, config: Dict[str, Any], output_path: Path) -> str:
        """Create distribution plot based on LLM configuration"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor='white')
            
            # Time series plot
            axes[0, 0].plot(data.index, data['value'], linewidth=2, color='#750014')
            axes[0, 0].set_title('Time Series')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_facecolor("white")
            axes[0, 0].grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.3)
        
            # Histogram with KDE
            sns.histplot(data['value'].dropna(), kde=True, ax=axes[0, 1], bins=30)
            axes[0, 1].set_title('Value Distribution')
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_facecolor("white")
            axes[0, 1].grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.3)
            
            # Box plot
            axes[1, 0].boxplot(data['value'].dropna())
            axes[1, 0].set_title('Value Box Plot')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_facecolor("white")
            axes[1, 0].grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.3)

            # Q-Q plot
            from scipy import stats
            stats.probplot(data['value'].dropna(), dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Q-Q Plot')
            axes[1, 1].set_facecolor("white")
            axes[1, 1].grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.3)

            plt.tight_layout()
            save_path = output_path / f"{config.get('name', 'data_distribution')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Distribution plot failed: {e}")
            return ""
    
    def _create_rolling_stats_plot(self, data: pd.DataFrame, config: Dict[str, Any], output_path: Path) -> str:
        """Create rolling statistics plot based on LLM configuration"""
        try:
            import matplotlib.pyplot as plt
            
            window_size = config.get('plot_specific_params', {}).get('window_size', 24)
            
            plt.figure(figsize=(12, 6), facecolor='white')
            
            for col in data.columns:
                rolling_mean = data[col].rolling(window=window_size).mean()
                rolling_std = data[col].rolling(window=window_size).std()
                
                plt.plot(data.index, rolling_mean, label=f'{col} - Rolling Mean', linewidth=2, color='#750014')
                plt.plot(data.index, rolling_std, label=f'{col} - Rolling Std', alpha=0.7, linewidth=2, color='#333333')
            
            # plt.title(config.get('title', f'Rolling Statistics (Window={window_size})'))
            plt.xlabel(config.get('xlabel', 'Time'))
            plt.ylabel(config.get('ylabel', 'Value'))
            plt.legend()
            # plt.grid(True)
            # 设置白色背景和黑色网格
            ax = plt.gca()
            ax.set_facecolor("white")
            ax.grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.3)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="black", alpha=0.2)

            # 增大字体
            plt.title(config.get('title', f'Rolling Statistics (Window={window_size})'), fontsize=16, fontweight='bold')
            plt.xlabel(config.get('xlabel', 'Time'), fontsize=14, fontweight='bold')
            plt.ylabel(config.get('ylabel', 'Value'), fontsize=14, fontweight='bold')
            
            # 增大刻度标签字体
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)
        
            
            save_path = output_path / f"{config.get('name', 'rolling_stats')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Rolling stats plot failed: {e}")
            return ""
    
    def _create_autocorrelation_plot(self, data: pd.DataFrame, config: Dict[str, Any], output_path: Path) -> str:
        """Create autocorrelation plot based on LLM configuration"""
        try:
            import matplotlib.pyplot as plt
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor='white')
            
            for col in data.columns:
                plot_acf(data[col].dropna(), ax=axes[0], lags=40, title='', linewidth=2, color='#750014')
                plot_pacf(data[col].dropna(), ax=axes[1], lags=40, title='', linewidth=2, color='#750014')

                for ax in axes:
                    ax.set_facecolor("white")
                    ax.grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.3)
                    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="black", alpha=0.2)

                    ax.set_title(ax.get_title(), fontsize=16, fontweight='bold')
                    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
                    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
                    
                    ax.tick_params(axis='x', labelsize=12)
                    ax.tick_params(axis='y', labelsize=12)

                    for line in ax.get_lines():
                        line.set_color('#750014')
                    for collection in ax.collections:
                        collection.set_color('#750014')
                        collection.set_edgecolor('#750014')
        
            plt.tight_layout()
            save_path = output_path / f"{config.get('name', 'autocorrelation')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Autocorrelation plot failed: {e}")
            return ""
    
    def _create_seasonal_decomposition_plot(self, data: pd.DataFrame, config: Dict[str, Any], output_path: Path) -> str:
        """Create seasonal decomposition plot based on LLM configuration"""
        try:
            import matplotlib.pyplot as plt
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            period = config.get('plot_specific_params', {}).get('period', 12)
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), facecolor='white')
            
            for col in data.columns:
                decomposition = seasonal_decompose(data[col].dropna(), period=period, extrapolate_trend='freq')
                
                decomposition.observed.plot(ax=axes[0], linewidth=2, color='#750014')
                decomposition.trend.plot(ax=axes[1], linewidth=2, color='#750014')
                decomposition.seasonal.plot(ax=axes[2], linewidth=2, color='#750014')
                decomposition.resid.plot(ax=axes[3], linewidth=2, color='#750014')
                for ax in axes:
                    ax.set_facecolor("white")
                    ax.grid(True, which="major", linestyle="-", linewidth=0.5, color="black", alpha=0.3)
                    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="black", alpha=0.2)
                    ax.set_title(ax.get_title(), fontsize=16, fontweight='bold')
                    ax.set_xlabel(ax.get_xlabel(), fontsize=14, fontweight='bold')
                    ax.set_ylabel(ax.get_ylabel(), fontsize=14, fontweight='bold')
                    
                    ax.tick_params(axis='x', labelsize=12)
                    ax.tick_params(axis='y', labelsize=12)
            
            plt.tight_layout()
            save_path = output_path / f"{config.get('name', 'seasonal_decomposition')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Seasonal decomposition plot failed: {e}")
            return ""
    
    def _plot_data_distribution(self, data: pd.DataFrame, save_path: str) -> str:
        """Plot data distribution - now uses LLM decisions"""
        # This method is kept for backward compatibility but now delegates to the new LLM-driven approach
        config = {
            "name": "distribution",
            "type": "distribution",
            "title": "Data Distribution Analysis"
        }
        output_path = Path(save_path).parent
        return self._create_distribution_plot(data, config, output_path)
    
    def _save_preprocess_results(self, data: pd.DataFrame, analysis_report: Dict[str, Any], output_dir: str):
        """Save preprocessing results"""
        logger.info("Saving preprocessing results...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned data
        data_path = output_path / "cleaned_data.csv"
        data.to_csv(data_path)
        logger.info(f"Cleaned data saved to {data_path}")
        
        # Save analysis report
        from utils.file_utils import FileSaver
        report_path = output_path / "analysis_report.json"
        FileSaver.save_json(analysis_report, report_path)
        logger.info(f"Analysis report saved to {report_path}")
        
        # Save preprocess summary
        summary = {
            'data_shape': data.shape,
            'data_columns': list(data.columns),
            'data_types': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'basic_stats': {
                'mean': data['value'].mean(),
                'std': data['value'].std(),
                'min': data['value'].min(),
                'max': data['value'].max(),
                'median': data['value'].median()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = output_path / "preprocess_summary.json"
        FileSaver.save_json(summary, summary_path)
        logger.info(f"Preprocess summary saved to {summary_path}")
    
    def _update_memory(self, data: pd.DataFrame, analysis_report: Dict[str, Any], visualizations: Dict[str, str]):
        """Update memory"""
        self.memory.store('cleaned_data', data, 'data')
        self.memory.store('analysis_report', analysis_report, 'analysis')
        self.memory.store('preprocess_visualizations', visualizations, 'visualizations')
        self.memory.store('preprocess_config', self.preprocess_config, 'config')
        
        # Record preprocessing history
        self.memory.add_history(
            'preprocess',
            {
                'data_shape': data.shape,
                'quality_score': analysis_report.get('quality_assessment', {}).get('data_quality_score', 0),
                'visualization_count': len(visualizations)
            }
        )
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get preprocessing summary"""
        analysis_report = self.memory.retrieve('analysis_report', 'analysis')
        if not analysis_report:
            return {}
        
        return {
            'data_shape': self.memory.retrieve('cleaned_data', 'data').shape if self.memory.retrieve('cleaned_data', 'data') is not None else None,
            'quality_score': analysis_report.get('quality_assessment', {}).get('data_quality_score', 0),
            'data_characteristics': analysis_report.get('quality_assessment', {}).get('data_characteristics', ''),
            'forecasting_readiness': analysis_report.get('forecasting_readiness', {}).get('data_suitability', ''),
            'preprocess_config': self.memory.retrieve('preprocess_config', 'config')
        } 
