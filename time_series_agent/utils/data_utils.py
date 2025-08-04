"""
Data Utilities for Time Series Prediction Agent
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data Loader"""
    
    @staticmethod
    def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        try:
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"Successfully loaded CSV file: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV file {filepath}: {e}")
            raise
    
    @staticmethod
    def load_json(filepath: str, **kwargs) -> pd.DataFrame:
        """Load JSON file"""
        try:
            df = pd.read_json(filepath, **kwargs)
            logger.info(f"Successfully loaded JSON file: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load JSON file {filepath}: {e}")
            raise
    
    @staticmethod
    def load_excel(filepath: str, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        try:
            df = pd.read_excel(filepath, **kwargs)
            logger.info(f"Successfully loaded Excel file: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Failed to load Excel file {filepath}: {e}")
            raise
    
    @staticmethod
    def load_data(filepath: str, **kwargs) -> pd.DataFrame:
        """Generic data loader"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        file_extension = filepath.suffix.lower()
        
        if file_extension == '.csv':
            return DataLoader.load_csv(filepath, **kwargs)
        elif file_extension == '.json':
            return DataLoader.load_json(filepath, **kwargs)
        elif file_extension in ['.xlsx', '.xls']:
            return DataLoader.load_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")


class DataPreprocessor:
    """Data Preprocessor"""
    
    @staticmethod
    def convert_to_time_series(df: pd.DataFrame, 
                             date_column: str = 'date',
                             value_column: str = 'OT',
                             **kwargs) -> pd.DataFrame:
        """Convert to time series format"""
        try:
            df_ts = df.copy()
            if date_column not in df_ts.columns:
                raise ValueError(f"Date column '{date_column}' not found in dataframe")
            
            df_ts[date_column] = pd.to_datetime(df_ts[date_column], **kwargs)
            df_ts.set_index(date_column, inplace=True)
            
            if value_column not in df_ts.columns:
                numeric_columns = df_ts.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    value_column = numeric_columns[0]
                    logger.warning(f"Value column not found, using first numeric column: {value_column}")
                else:
                    raise ValueError(f"Value column '{value_column}' not found and no numeric columns available")
            
            df_ts = df_ts[[value_column]]
            df_ts.columns = ['value']
            df_ts.sort_index(inplace=True)
            
            logger.info(f"Successfully converted to time series format with {len(df_ts)} data points")
            return df_ts
            
        except Exception as e:
            logger.error(f"Failed to convert to time series format: {e}")
            raise
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, 
                            strategy: str = 'interpolate',
                            **kwargs) -> pd.DataFrame:
        """Handle missing values"""
        df_clean = df.copy()
        
        if df_clean.isnull().sum().sum() == 0:
            logger.info("No missing values found")
            return df_clean
        
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        if strategy == 'interpolate':
            df_clean = df_clean.interpolate(**kwargs)
        elif strategy == 'forward_fill':
            df_clean = df_clean.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df_clean = df_clean.fillna(method='bfill')
        elif strategy == 'mean':
            df_clean = df_clean.fillna(df_clean.mean())
        elif strategy == 'median':
            df_clean = df_clean.fillna(df_clean.median())
        elif strategy == 'drop':
            df_clean = df_clean.dropna()
        elif strategy == 'zero':
            df_clean = df_clean.fillna(0)
        elif strategy == 'none':
            pass
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")
        
        remaining_missing = df_clean.isnull().sum().sum()
        logger.info(f"Missing values handled. Remaining missing values: {remaining_missing}")
        
        return df_clean
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, 
                       method: str = 'iqr',
                       threshold: float = 1.5,
                       window_size: int = 24,
                       **kwargs) -> Dict[str, List[int]]:
        """Detect outliers. For IQR, use rolling window (default window_size=24)."""
        outliers = {}
        for column in df.columns:
            if df[column].dtype in [np.number]:
                if method == 'iqr':
                    series = df[column]
                    outlier_indices = []
                    for i in range(len(series)):
                        # Rolling window
                        start = max(0, i - window_size // 2)
                        end = min(len(series), i + window_size // 2 + 1)
                        window = series.iloc[start:end]
                        Q1 = window.quantile(0.25)
                        Q3 = window.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        if series.iloc[i] < lower_bound or series.iloc[i] > upper_bound:
                            outlier_indices.append(series.index[i])
                    outliers[column] = outlier_indices
                elif method == 'zscore':
                    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                    outlier_indices = df[z_scores > threshold].index.tolist()
                    outliers[column] = outlier_indices
                elif method == 'percentile':
                    lower_percentile = kwargs.get('lower_percentile', 1)
                    upper_percentile = kwargs.get('upper_percentile', 99)
                    lower_bound = df[column].quantile(lower_percentile / 100)
                    upper_bound = df[column].quantile(upper_percentile / 100)
                    outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
                    outliers[column] = outlier_indices
                elif method == 'none':
                    pass
                else:
                    raise ValueError(f"Unknown outlier detection method: {method}")
        total_outliers = sum(len(indices) for indices in outliers.values())
        logger.info(f"Detected {total_outliers} outliers using {method} method (window_size={window_size})")
        return outliers
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, 
                       outliers: Dict[str, List[int]],
                       strategy: str = 'clip',
                       **kwargs) -> pd.DataFrame:
        """
        Handle outliers in a time series DataFrame.
        Supported strategies:
            - 'clip' (default): clip outlier values to non-outlier min/max
            - 'drop': drop outlier indices
            - 'interpolate': linear interpolate outlier indices
            - 'ffill': forward fill outlier indices
            - 'bfill': backward fill outlier indices
            - 'mean': replace outlier with mean of non-outliers
            - 'median': replace outlier with median of non-outliers
            - 'smooth': moving average filter (window_size)
        """
        import numpy as np
        df_clean = df.copy()
        for column, outlier_indices in outliers.items():
            if not outlier_indices:
                continue
            all_indices = set(df_clean.index)
            outlier_set = set(outlier_indices)
            non_outlier_indices = list(all_indices - outlier_set)
            if strategy == 'clip':
                # Find non-outlier min/max as boundaries
                if non_outlier_indices:
                    non_outlier_values = df_clean.loc[non_outlier_indices, column]
                    min_val = non_outlier_values.min()
                    max_val = non_outlier_values.max()
                else:
                    min_val = df_clean[column].min()
                    max_val = df_clean[column].max()
                for idx in outlier_indices:
                    val = df_clean.at[idx, column]
                    if val < min_val:
                        df_clean.at[idx, column] = min_val
                    elif val > max_val:
                        df_clean.at[idx, column] = max_val
            elif strategy == 'drop':
                df_clean = df_clean.drop(outlier_indices)
            elif strategy == 'interpolate':
                df_clean.loc[outlier_indices, column] = np.nan
                df_clean[column] = df_clean[column].interpolate(method='linear')
            elif strategy == 'ffill':
                df_clean.loc[outlier_indices, column] = np.nan
                df_clean[column] = df_clean[column].fillna(method='ffill')
            elif strategy == 'bfill':
                df_clean.loc[outlier_indices, column] = np.nan
                df_clean[column] = df_clean[column].fillna(method='bfill')
            elif strategy == 'mean':
                if non_outlier_indices:
                    mean_val = df_clean.loc[non_outlier_indices, column].mean()
                else:
                    mean_val = df_clean[column].mean()
                for idx in outlier_indices:
                    df_clean.at[idx, column] = mean_val
            elif strategy == 'median':
                if non_outlier_indices:
                    median_val = df_clean.loc[non_outlier_indices, column].median()
                else:
                    median_val = df_clean[column].median()
                for idx in outlier_indices:
                    df_clean.at[idx, column] = median_val
            elif strategy == 'smooth':
                window = kwargs.get('window', 5)
                smoothed = df_clean[column].rolling(window=window, center=True, min_periods=1).mean()
                for idx in outlier_indices:
                    df_clean.at[idx, column] = smoothed.at[idx]
            else:
                raise ValueError(f"Unknown outlier handling strategy: {strategy}")
        logger.info(f"Handled outliers using {strategy} strategy")
        return df_clean


class DataSplitter:
    """Data Splitter for agent framework (validation + test only, no train set)"""
    
    @staticmethod
    def create_slices(df: pd.DataFrame,
                     num_slices: int,
                     input_length: int,
                     horizon: int,
                     **kwargs) -> List[Dict[str, pd.DataFrame]]:
        """
        Create data slices for agent framework.
        Each slice contains only validation (input_length) and test (horizon), no train set.
        Slices cover as much data as possible.
        """
        slices = []
        total_length = len(df)
        min_required = input_length + horizon
        if total_length < min_required:
            raise ValueError(f"Not enough data to create a single slice: need at least {min_required} rows.")
        
        # Compute stride so that slices are as evenly distributed as possible
        stride = (total_length - min_required) // max(num_slices - 1, 1) if num_slices > 1 else 0
        for i in range(num_slices):
            validation_start = i * stride
            validation_end = validation_start + input_length
            test_start = validation_end
            test_end = test_start + horizon
            if test_end > total_length:
                break  # Not enough data for a full slice
            validation_data = df.iloc[validation_start:validation_end]
            test_data = df.iloc[test_start:test_end]
            slices.append({
                'validation': validation_data,
                'test': test_data,
                'slice_id': i,
                'validation_start': validation_start,
                'validation_end': validation_end,
                'test_start': test_start,
                'test_end': test_end,
            })
        logger.info(f"Created {len(slices)} agent slices")
        return slices


class DataValidator:
    """Data Validator"""
    
    @staticmethod
    def validate_time_series(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Validate time series data"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # check if index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            validation_results['is_valid'] = False
            validation_results['errors'].append("Index is not a DatetimeIndex")
        
        # check for duplicate timestamps
        if df.index.duplicated().any():
            validation_results['warnings'].append("Duplicate timestamps found")
        
        # check if data is sorted by time
        if not df.index.is_monotonic_increasing:
            validation_results['warnings'].append("Data is not sorted by time")
        
        # check missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            validation_results['warnings'].append(f"Found {missing_count} missing values")
        
        # check data length
        if len(df) < 10:
            validation_results['warnings'].append("Data length is very short (< 10)")
        
        # check numeric range
        for column in df.columns:
            if df[column].dtype in [np.number]:
                if df[column].isnull().all():
                    validation_results['errors'].append(f"Column {column} contains only null values")
                elif df[column].std() == 0:
                    validation_results['warnings'].append(f"Column {column} has zero variance")
        
        # update validation status
        if validation_results['errors']:
            validation_results['is_valid'] = False
        
        # add basic information
        validation_results['info'] = {
            'length': len(df),
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'date_range': {
                'start': df.index.min().isoformat() if len(df) > 0 else None,
                'end': df.index.max().isoformat() if len(df) > 0 else None
            }
        }
        
        return validation_results
    
    @staticmethod
    def check_stationarity(df: pd.DataFrame, 
                          column: str = 'value',
                          **kwargs) -> Dict[str, Any]:
        """Check time series stationarity"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            series = df[column].dropna()
            
            if len(series) < 10:
                return {
                    'is_stationary': False,
                    'p_value': None,
                    'test_statistic': None,
                    'error': 'Insufficient data for stationarity test'
                }
            
            # perform ADF test
            result = adfuller(series, **kwargs)
            
            return {
                'is_stationary': result[1] < 0.05,
                'p_value': result[1],
                'test_statistic': result[0],
                'critical_values': result[4]
            }
            
        except ImportError:
            logger.warning("statsmodels not available, skipping stationarity test")
            return {
                'is_stationary': None,
                'p_value': None,
                'test_statistic': None,
                'error': 'statsmodels not available'
            }
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return {
                'is_stationary': None,
                'p_value': None,
                'test_statistic': None,
                'error': str(e)
            }


class DataAnalyzer:
    """Data Analyzer"""
    
    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics"""
        stats = {}
        
        for column in df.columns:
            if df[column].dtype in [np.number]:
                col_stats = df[column].describe()
                stats[column] = {
                    'count': col_stats['count'],
                    'mean': col_stats['mean'],
                    'std': col_stats['std'],
                    'min': col_stats['min'],
                    '25%': col_stats['25%'],
                    '50%': col_stats['50%'],
                    '75%': col_stats['75%'],
                    'max': col_stats['max'],
                    'skewness': df[column].skew(),
                    'kurtosis': df[column].kurtosis()
                }
        
        return stats
    
    @staticmethod
    def detect_seasonality(df: pd.DataFrame,
                          column: str = 'value',
                          max_lag: int = 50,
                          **kwargs) -> Dict[str, Any]:
        """Detect seasonality"""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            series = df[column].dropna()
            
            if len(series) < 2 * max_lag:
                return {
                    'has_seasonality': False,
                    'seasonal_period': None,
                    'error': 'Insufficient data for seasonality detection'
                }
            
            # try different periods for decomposition
            best_period = None
            best_seasonal_strength = 0
            
            for period in range(2, min(max_lag, len(series) // 2)):
                try:
                    decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
                    seasonal_strength = decomposition.seasonal.var() / series.var()
                    
                    if seasonal_strength > best_seasonal_strength:
                        best_seasonal_strength = seasonal_strength
                        best_period = period
                        
                except:
                    continue
            
            return {
                'has_seasonality': best_seasonal_strength > 0.1,
                'seasonal_period': best_period,
                'seasonal_strength': best_seasonal_strength
            }
            
        except ImportError:
            logger.warning("statsmodels not available, skipping seasonality detection")
            return {
                'has_seasonality': None,
                'seasonal_period': None,
                'error': 'statsmodels not available'
            }
        except Exception as e:
            logger.error(f"Error in seasonality detection: {e}")
            return {
                'has_seasonality': None,
                'seasonal_period': None,
                'error': str(e)
            }
    
    @staticmethod
    def analyze_trends(df: pd.DataFrame,
                      column: str = 'value',
                      **kwargs) -> Dict[str, Any]:
        """Analyze trends"""
        series = df[column].dropna()
        
        if len(series) < 2:
            return {
                'has_trend': False,
                'trend_direction': None,
                'trend_strength': None,
                'error': 'Insufficient data for trend analysis'
            }
        
        # calculate linear trend
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series, 1)
        
        # calculate trend strength
        trend_strength = abs(slope) / series.std()
        
        return {
            'has_trend': trend_strength > 0.01,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing',
            'trend_strength': trend_strength,
            'slope': slope,
            'intercept': intercept
        }


def load_and_preprocess_data(filepath: str,
                           config: Dict[str, Any],
                           **kwargs) -> pd.DataFrame:
    """Convenient function to load and preprocess data"""
    df = DataLoader.load_data(filepath, **kwargs)
    
    # convert to time series format
    date_column = config.get('date_column', 'date')
    value_column = config.get('value_column', 'value')
    df_ts = DataPreprocessor.convert_to_time_series(df, date_column, value_column)
    
    # handle missing values
    missing_strategy = config.get('missing_value_strategy', 'interpolate')
    df_ts = DataPreprocessor.handle_missing_values(df_ts, strategy=missing_strategy)
    
    # detect and handle outliers
    outlier_method = config.get('outlier_method', 'iqr')
    outlier_window_size = config.get('outlier_window_size', 24) # Add window_size to config
    outliers = DataPreprocessor.detect_outliers(df_ts, method=outlier_method, window_size=outlier_window_size)
    outlier_strategy = config.get('outlier_strategy', 'clip')
    df_ts = DataPreprocessor.handle_outliers(df_ts, outliers, strategy=outlier_strategy)
    
    return df_ts 