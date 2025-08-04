"""
Default Configuration for Time Series Prediction Agent
"""

import os
from typing import Dict, Any, List

# LLM configuration
LLM_CONFIG = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 4000,
        "api_base": "https://api.openai.com/v1"
    },
    "google": {
        "provider": "google",
        "model": "gemini-2.0-flash",
        "temperature": 0.1,
        "max_tokens": 4000,
        "api_base": "https://generativelanguage.googleapis.com/v1"
    },
    "anthropic": {
        "provider": "anthropic", 
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.1,
        "max_tokens": 4000,
        "api_base": "https://api.anthropic.com"
    }
}

# Data preprocessing configuration
PREPROCESS_CONFIG = {
    "missing_value_strategy": "interpolate",  # interpolate, forward_fill, backward_fill, mean, median, drop, zero
    "outlier_strategy": "clip",  # clip, drop, iqr
    "normalization": False,  # 是否进行归一化
    "scaler_type": "dummy",  # minmax, standard, quantile, log, dummy
    "visualization": True,  # 是否生成可视化
    "save_intermediate": True  # 是否保存中间结果
}

# Model configuration
MODEL_CONFIG = {
    "available_models": [
        "ARIMA", "RandomWalk", "ExponentialSmoothing", "MovingAverage", "LinearRegression", 
        "PolynomialRegression", "RidgeRegression", "LassoRegression", "ElasticNet", "SVR", 
        "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", "NeuralNetwork", "LSTM", 
        "Prophet", "TBATS", "Theta", "Croston", "Transformer"
    ],
    "k_models": 3,
    "n_candidates": 5,
    "ensemble_method": "weighted_average",  # simple_average, weighted_average, trimmed_mean, median
    "hyperparameter_optimization": True,
    "cross_validation_folds": 3
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    "num_slices": 10,  # 数据切片数量
    "input_length": 512,  # 输入序列长度
    "horizon": 96,  # 预测步长
    "validation_ratio": 0.2,  # 验证集比例
    "test_ratio": 0.2,  # 测试集比例
    "random_seed": 42,
    "parallel_processing": False,  # 是否并行处理
    "max_workers": 4
}

# Evaluation metrics configuration
METRICS_CONFIG = {
    "primary_metric": "mse",  # mse, mae, mape
    "secondary_metrics": ["mae", "mape"],
    "confidence_level": 0.95,
    "bootstrap_samples": 1000
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "style": "seaborn-v0_8",
    "color_palette": "husl",
    "save_format": "png",
    "show_plots": False
}

# Report configuration
REPORT_CONFIG = {
    "generate_comprehensive_report": True,
    "include_confidence_intervals": True,
    "include_model_interpretability": True,
    "include_workflow_documentation": True,
    "report_format": "json",  # json, html, pdf
    "save_individual_reports": True
}

# System configuration
SYSTEM_CONFIG = {
    "debug": False,
    "verbose": False,
    "log_level": "INFO",
    "save_logs": True,
    "max_memory_usage": "8GB",
    "timeout": 3600,  # 超时时间（秒）
    "retry_attempts": 3
}

# Path configuration
PATH_CONFIG = {
    "output_dir": "results",
    "log_dir": "logs",
    "cache_dir": "cache",
    "model_dir": "models",
    "visualization_dir": "visualizations",
    "report_dir": "reports"
}

# Default configuration
DEFAULT_CONFIG = {
    # Basic configuration
    "data_path": None,  # need to be specified by user
    "output_dir": "results_ETTh2_720",
    
    # LLM configuration
    "llm_provider": "openai",
    "llm_model": "gpt-4o",
    "llm_temperature": 0.1,
    "llm_max_tokens": 4000,
    
    # Experiment parameters
    "num_slices": 10,
    "input_length": 512,
    "horizon": 96,
    "k_models": 3,
    
    # System parameters
    "debug": False,
    "verbose": False,
    "random_seed": 42,
    
    # Merge all sub-configurations
    "preprocess": PREPROCESS_CONFIG,
    "models": MODEL_CONFIG,
    "experiment": EXPERIMENT_CONFIG,
    "metrics": METRICS_CONFIG,
    "visualization": VISUALIZATION_CONFIG,
    "report": REPORT_CONFIG,
    "system": SYSTEM_CONFIG,
    "paths": PATH_CONFIG
}

# Environment variable configuration
ENV_VARS = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
    "GOOGLE_API_BASE": os.getenv("GOOGLE_API_BASE"),
    "ANTHROPIC_API_BASE": os.getenv("ANTHROPIC_API_BASE")
}

# Model hyperparameters configuration
MODEL_HYPERPARAMETERS = {
    "ARMA": {
        "p": [1, 2, 3],
        "q": [1, 2, 3],
        "d": [0, 1]
    },
    "LSTM": {
        "units": [50, 100, 200],
        "layers": [1, 2, 3],
        "dropout": [0.1, 0.2, 0.3],
        "batch_size": [32, 64, 128],
        "epochs": [50, 100, 200]
    },
    "ExponentialSmoothing": {
        "trend": ["add", "mul", None],
        "seasonal": ["add", "mul", None],
        "seasonal_periods": [12, 24, 48]
    },
    "RandomForest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200, 500],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3]
    },
    "SVR": {
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"]
    }
}

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    "missing_ratio_threshold": 0.3,
    "outlier_ratio_threshold": 0.1,
    "min_data_points": 100, 
    "max_data_points": 100000, 
    "stationarity_p_value": 0.05
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "min_mape": 0.01,
    "max_mape": 100.0,
    "min_mse": 1e-6, 
    "max_mse": 1e6,
    "convergence_tolerance": 1e-6 
}

# Extended configuration
DEFAULT_CONFIG.update({
    "hyperparameters": MODEL_HYPERPARAMETERS,
    "data_quality": DATA_QUALITY_THRESHOLDS,
    "performance": PERFORMANCE_THRESHOLDS,
    "env_vars": ENV_VARS
})


def get_llm_config(provider: str) -> Dict[str, Any]:
    """Get LLM configuration for a specified provider"""
    return LLM_CONFIG.get(provider, LLM_CONFIG["openai"])


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the configuration for validity"""
    validated_config = config.copy()
    
    # Validate required parameters
    if not config.get("data_path"):
        raise ValueError("data_path is a required configuration parameter")
    
    # Validate numerical ranges
    if config.get("num_slices", 0) <= 0:
        validated_config["num_slices"] = 10
    
    if config.get("input_length", 0) <= 0:
        validated_config["input_length"] = 512
    
    if config.get("horizon", 0) <= 0:
        validated_config["horizon"] = 96
    
    if config.get("k_models", 0) <= 0:
        validated_config["k_models"] = 3
    
    # Validate LLM configuration
    provider = config.get("llm_provider", "openai")
    if provider not in LLM_CONFIG:
        validated_config["llm_provider"] = "openai"
    
    return validated_config


def create_config_from_args(**kwargs) -> Dict[str, Any]:
    """Create configuration from arguments"""
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return validate_config(config) 