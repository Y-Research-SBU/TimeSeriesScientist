<div align="center">

<h2>Time Series Prediction Agent</h2>
</div>

This is the repo for time series forecasting agent system - Intelligent time series prediction framework based on LangGraph

## Project Overview


Time Series Prediction Agent is a time series forecasting system based on LangGraph, using a multi-agent collaborative approach to implement a complete workflow from data preprocessing to final prediction reports.
![workflow](/assets/workflow.jpg)
## Environment Setup

### 1. Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Sufficient disk space for results and models

### 2. Create Virtual Environment

```bash
# Create virtual environment
conda create -n TimeAgent

conda activate TimeAgent
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Optional: Set other environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Code Execution

### 1. Basic Usage

```bash
# Run with default configuration
python main.py
```

### 2. Custom Configuration

```bash
# Run with custom parameters
python main.py --data_path /your-customized-dataset \
                --num_slices 10 \
                --horizon 96 \
                --llm_model gpt-4o
```

### 3. Configuration File

Create a custom configuration file `config/my_config.yaml`:

```yaml
# Data configuration
data_path: "../dataset/ETT-small/ETTh1.csv"
date_column: "date"
value_column: "OT"

# Experiment parameters
num_slices: 10
input_length: 512
horizon: 96
k_models: 3

# LLM configuration
llm_provider: "openai"
llm_model: "gpt-4o"
llm_temperature: 0.1
```

Then run:

```bash
python main.py --config_file config/my_config.yaml
```

### 4. Debug Mode

```bash
# Run with debug and verbose output
python main.py --debug --verbose --data_path your_data.csv
```

## Agent Workflow

The system uses a sophisticated multi-agent workflow orchestrated by LangGraph. Here's how the agents work together:

### 1. PreprocessAgent

**Purpose**: Data cleaning and preparation

**Responsibilities**:
- Load and validate time series data
- Handle missing values using LLM-recommended strategies
- Detect and handle outliers using rolling window IQR
- Generate data quality reports
- Create preprocessing visualizations

**LLM Integration**: Uses LLM to analyze data quality and recommend preprocessing strategies

**Output**: Cleaned data, quality metrics, and preprocessing visualizations

### 2. AnalysisAgent

**Purpose**: Comprehensive data analysis and insights

**Responsibilities**:
- Analyze data characteristics (trend, seasonality, stationarity)
- Generate statistical summaries
- Identify patterns and anomalies
- Provide forecasting readiness assessment
- Create analysis visualizations

**LLM Integration**: Uses LLM to interpret data patterns and provide insights

**Output**: Analysis report with key findings and recommendations

### 3. ValidationAgent

**Purpose**: Model selection and hyperparameter optimization

**Responsibilities**:
- Select best models based on data characteristics
- Optimize hyperparameters for each selected model
- Evaluate models on validation data
- Rank models by performance
- Generate validation reports

**LLM Integration**: Uses LLM to select appropriate models and hyperparameters

**Output**: Selected models with optimized hyperparameters and validation scores

### 4. ForecastAgent

**Purpose**: Generate predictions and ensemble forecasts

**Responsibilities**:
- Train models with optimized hyperparameters
- Generate individual model predictions
- Create ensemble predictions using multiple methods
- Calculate confidence intervals
- Generate forecast visualizations
- Compute performance metrics

**LLM Integration**: Uses LLM to decide ensemble integration strategies

**Output**: Individual and ensemble predictions, metrics, and visualizations

### 5. ReportAgent

**Purpose**: Generate comprehensive final reports

**Responsibilities**:
- Synthesize all experiment results
- Create executive summaries
- Generate actionable recommendations
- Produce final visualizations
- Save results in multiple formats

**LLM Integration**: Uses LLM to generate human-readable reports

**Output**: Comprehensive experiment report with insights and recommendations

## Workflow Execution

```
Data Input → PreprocessAgent → AnalysisAgent → ValidationAgent → ForecastAgent → ReportAgent → Final Output
     ↓              ↓              ↓              ↓              ↓              ↓
  Raw Data    Cleaned Data   Analysis Report   Selected Models   Predictions   Final Report
```

### Key Features of the Workflow:

1. **State Management**: Each agent receives and updates a shared state object
2. **Error Recovery**: Built-in retry mechanisms and fallback strategies
3. **Rate Limiting**: Automatic delays and retries for API rate limits
4. **Progress Monitoring**: Real-time progress tracking and timing information
5. **Result Aggregation**: Automatic aggregation of results across multiple data slices

## Project Structure

```
time_series_agent/
│
├── agents/                    # Agent modules
│   ├── preprocess_agent.py    # Data preprocessing agent
│   ├── analysis_agent.py      # Data analysis agent
│   ├── validation_agent.py    # Model validation agent
│   ├── forecast_agent.py      # Forecasting agent
│   ├── report_agent.py        # Report generation agent
│   └── memory.py              # Memory management module
│
├── graph/                     # Workflow orchestration
│   └── agent_graph.py         # LangGraph workflow definition
│
├── utils/                     # Utility modules
│   ├── data_utils.py          # Data utilities
│   ├── visualization_utils.py # Visualization utilities
│   ├── file_utils.py          # File management utilities
│   └── model_library.py       # Model implementations
│
├── config/                    # Configuration
│   └── default_config.py      # Default configuration
│
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Output Structure

```
results/
├── reports/                   # Generated reports
│   ├── complete_time_series_report_YYYYMMDD_HHMMSS.json
│   └── aggregated_forecast_results_YYYYMMDD_HHMMSS.json
├── preprocess/                # Preprocessing outputs
│   ├── visualizations/
│   └── analysis_reports/
├── forecast/                  # Forecasting outputs
│   ├── visualizations/
│   └── prediction_results/
└── logs/                      # Execution logs
```

## Supported Models

### Statistical Models
- ARMA, ARIMA
- Exponential Smoothing
- TBATS, Theta
- Prophet

### Machine Learning Models
- Linear Regression
- Random Forest
- SVR (Support Vector Regression)
- Gradient Boosting
- XGBoost, LightGBM

### Deep Learning Models
- LSTM
- Neural Networks
- Transformer

### Ensemble Methods
- Simple Average
- Weighted Average
- Median
- Trimmed Mean

## Configuration Options

### Basic Configuration

```python
config = {
    "data_path": "../dataset/ETT-small/ETTh1.csv",
    "num_slices": 10,
    "input_length": 512,
    "horizon": 96,
    "k_models": 3,
    "llm_model": "gpt-4o",
    "debug": True
}
```

### Advanced Configuration

```python
config = {
    # Data processing
    "preprocess": {
        "missing_value_strategy": "interpolate",
        "outlier_strategy": "clip",
        "normalization": False
    },
    
    # Model selection
    "models": {
        "available_models": ["ARMA", "LSTM", "RandomForest"],
        "ensemble_method": "weighted_average",
        "hyperparameter_optimization": True
    },
    
    # Visualization
    "visualization": {
        "figure_size": (12, 8),
        "save_format": "png",
        "show_plots": False
    }
}
```

## Troubleshooting

### Common Issues

1. **API Rate Limiting**
   - The system automatically handles rate limits with retries
   - Increase delays between slices if needed
   - Check your OpenAI API usage limits

2. **Memory Issues**
   - Reduce `num_slices` parameter
   - Use smaller `input_length`
   - Monitor system memory usage

3. **Model Import Errors**
   - Install missing dependencies: `pip install prophet tbats`
   - Some models have optional dependencies
   - Check the model library for requirements

### Debug Mode

Enable debug mode for detailed logging:

```bash
python main.py --debug --verbose
```

## Performance Optimization

### For Large Datasets
- Reduce `num_slices` to process fewer slices
- Use smaller `input_length` for faster processing
- Enable parallel processing in configuration

### For API Rate Limits
- Increase delays between agent calls
- Use fallback strategies when API fails
- Monitor API usage and adjust accordingly

## License

MIT License
