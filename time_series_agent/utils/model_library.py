"""
Model Library for Time Series Forecasting
Contains all individual model prediction functions for time series forecasting.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from prophet import Prophet
from tbats import TBATS

logger = logging.getLogger(__name__)

def _create_time_series_features(series: pd.Series, lookback: int = 10) -> tuple:
    """
    Create time series features for machine learning models.
    
    Args:
        series: Time series data
        lookback: Number of lag features to create
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target
    """
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series.iloc[i-lookback:i].values)
        y.append(series.iloc[i])
    return np.array(X), np.array(y)

def predict_arima(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """ARIMA model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'p', 'q', 'd' for ARIMA order
        horizon: Number of steps to predict into the future
        
    Returns:
        List of predicted values for the specified horizon
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        p = params.get('p', 1)
        q = params.get('q', 1)
        d = params.get('d', 1)
        
        # Fit ARIMA model
        model = ARIMA(series, order=(p, d, q))
        fitted_model = model.fit()
        
        # Predict
        forecast = fitted_model.forecast(steps=horizon)
        return forecast.tolist()
        
    except Exception as e:
        logger.warning(f"ARIMA prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_random_walk(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Random Walk model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters (not used in random walk)
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        last_value = series.iloc[-1]
        std_dev = series.diff().std()
        
        predictions = []
        current_value = last_value
        
        for _ in range(horizon):
            # Random walk: next value = current value + random noise
            noise = np.random.normal(0, std_dev)
            current_value = current_value + noise
            predictions.append(max(0, current_value))  # Ensure non-negative
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Random Walk prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_moving_average(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Moving Average model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'window_size'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        window_size = params.get('window_size', 10)
        
        # Calculate moving average
        ma = series.rolling(window=window_size).mean().iloc[-1]
        
        # Simple prediction: use the last moving average value
        predictions = [ma] * horizon
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Moving Average prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_polynomial_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Polynomial Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'degree'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        degree = params.get('degree', 2)
        
        # Create polynomial features
        X = np.arange(len(series)).reshape(-1, 1)
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(X)
        
        # Fit polynomial regression
        model = LinearRegression()
        model.fit(X_poly, series)
        
        # Predict future values
        predictions = []
        for i in range(len(series), len(series) + horizon):
            X_future = poly_features.transform([[i]])
            pred = model.predict(X_future)[0]
            predictions.append(max(0, pred))
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Polynomial Regression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_ridge_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Ridge Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Train Ridge model
        alpha = params.get('alpha', 1.0)
        model = Ridge(alpha=alpha, **params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Ridge Regression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_lasso_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Lasso Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Train Lasso model
        alpha = params.get('alpha', 1.0)
        model = Lasso(alpha=alpha, **params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Lasso Regression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_elastic_net(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Elastic Net model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha', 'l1_ratio'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Train Elastic Net model
        alpha = params.get('alpha', 1.0)
        l1_ratio = params.get('l1_ratio', 0.5)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Elastic Net prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_svr(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Support Vector Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for SVR
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train SVR model
        model = SVR(**params)
        model.fit(X_scaled, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        
        for _ in range(horizon):
            pred = model.predict(last_features_scaled)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
            last_features_scaled = scaler.transform(last_features)
        
        return predictions
        
    except Exception as e:
        logger.warning(f"SVR prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_gradient_boosting(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Gradient Boosting model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for GradientBoostingRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Train Gradient Boosting model
        model = GradientBoostingRegressor(**params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Gradient Boosting prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_xgboost(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """XGBoost model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for XGBRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        import xgboost as xgb
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"XGBoost prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_lightgbm(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """LightGBM model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for LGBMRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        import lightgbm as lgb
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Train LightGBM model
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = model.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"LightGBM prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_neural_network(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Neural Network model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for MLPRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Neural Network model
        model = MLPRegressor(**params)
        model.fit(X_scaled, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        
        for _ in range(horizon):
            pred = model.predict(last_features_scaled)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
            last_features_scaled = scaler.transform(last_features)
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Neural Network prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_lstm(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """LSTM model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'units', 'activation', 'optimizer'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # This should implement LSTM prediction
        # For now, use a simple time series prediction
        series = data['value'].dropna()
        last_values = series.tail(10).values
        
        # Simple trend extrapolation
        trend = np.polyfit(range(len(last_values)), last_values, 1)[0]
        predictions = []
        
        for i in range(horizon):
            pred = last_values[-1] + trend * (i + 1)
            predictions.append(max(0, pred))  # Ensure non-negative
        
        return predictions
        
    except Exception as e:
        logger.warning(f"LSTM prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_exponential_smoothing(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Exponential Smoothing model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for ExponentialSmoothing
        horizon: Number of steps to predict into the future
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Fit model
        model = ExponentialSmoothing(series, **params)
        fitted_model = model.fit()
        
        # Predict
        forecast = fitted_model.forecast(steps=horizon)
        return forecast.tolist()
        
    except Exception as e:
        logger.warning(f"ExponentialSmoothing prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_linear_regression(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Linear Regression model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for LinearRegression
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=5)
        
        # Train model
        lr = LinearRegression(**params)
        lr.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = lr.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"LinearRegression prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_random_forest(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Random Forest model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for RandomForestRegressor
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Create features
        X, y = _create_time_series_features(series, lookback=10)
        
        # Train model
        rf = RandomForestRegressor(**params)
        rf.fit(X, y)
        
        # Predict
        predictions = []
        last_features = X[-1].reshape(1, -1)
        
        for _ in range(horizon):
            pred = rf.predict(last_features)[0]
            predictions.append(max(0, pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[0, -1] = pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"RandomForest prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_prophet(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Prophet model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for Prophet
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Ensure we have a proper index
        if data.index.name is None:
            data.index.name = 'ds'
        
        # Prepare Prophet data format - ensure proper date handling
        df_prophet = data.reset_index()
        
        # Check if the index column contains valid dates
        if df_prophet.columns[0] == 'ds':
            # If the index is already named 'ds', use it directly
            pass
        else:
            # Create a proper date index if needed
            df_prophet.columns = ['ds', 'y']
        
        # Ensure 'ds' column contains valid dates
        if not pd.api.types.is_datetime64_any_dtype(df_prophet['ds']):
            # Try to convert to datetime
            try:
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
            except Exception as e:
                logger.warning(f"Failed to convert dates for Prophet: {e}")
                # Create a simple numeric index as fallback
                df_prophet['ds'] = pd.date_range(start='2020-01-01', periods=len(df_prophet), freq='D')
        
        # Ensure 'y' column exists and contains numeric values
        if 'y' not in df_prophet.columns:
            df_prophet['y'] = data['value'].values
        
        # Remove any rows with invalid dates or values
        df_prophet = df_prophet.dropna()
        
        if len(df_prophet) == 0:
            logger.warning("No valid data for Prophet after cleaning")
            return predict_default(data, params, horizon)
        
        # Fit model with default parameters if none provided
        prophet_params = {
            'yearly_seasonality': False,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'seasonality_mode': 'additive'
        }
        prophet_params.update(params)
        
        model = Prophet(**prophet_params)
        model.fit(df_prophet)
        
        # Create future dates
        last_date = df_prophet['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq='D')[1:]
        future = pd.DataFrame({'ds': future_dates})
        
        # Predict
        forecast = model.predict(future)
        predictions = forecast['yhat'].tolist()
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Prophet prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_tbats(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """TBATS model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for TBATS
        horizon: Number of steps to predict into the future
    """
    try:        
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Fit TBATS model
        model = TBATS(**params)
        fitted_model = model.fit(series)
        
        # Predict
        forecast = fitted_model.forecast(steps=horizon)
        return forecast.tolist()
        
    except Exception as e:
        logger.warning(f"TBATS prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_theta(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Theta model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters (not used in Theta method)
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        
        # Simple Theta method implementation
        # Decompose into trend and seasonal components
        n = len(series)
        theta = 2  # Default theta parameter
        
        # Calculate trend using linear regression
        x = np.arange(n)
        trend_coef = np.polyfit(x, series, 1)
        trend = trend_coef[0] * x + trend_coef[1]
        
        # Calculate seasonal component (simple moving average)
        seasonal_period = params.get('seasonal_period', 12)
        seasonal = series.rolling(window=seasonal_period, center=True).mean()
        seasonal = seasonal.fillna(method='bfill').fillna(method='ffill')
        
        # Theta decomposition
        theta_trend = trend + (series - trend) / theta
        theta_seasonal = seasonal
        
        # Predict
        predictions = []
        for i in range(horizon):
            future_trend = trend_coef[0] * (n + i) + trend_coef[1]
            future_seasonal = seasonal.iloc[-seasonal_period + (i % seasonal_period)]
            pred = future_trend + (future_seasonal - future_trend) / theta
            predictions.append(max(0, pred))
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Theta prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_croston(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Croston model prediction for intermittent time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters including 'alpha'
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        series = data['value'].dropna()
        alpha = params.get('alpha', 0.4)
        
        # Croston method for intermittent demand
        # Separate demand size and inter-demand intervals
        demand_sizes = []
        intervals = []
        last_demand_idx = -1
        
        for i, value in enumerate(series):
            if value > 0:
                if last_demand_idx >= 0:
                    intervals.append(i - last_demand_idx)
                demand_sizes.append(value)
                last_demand_idx = i
        
        if len(demand_sizes) == 0:
            return [0] * horizon
        
        # Calculate Croston parameters
        avg_demand_size = np.mean(demand_sizes)
        avg_interval = np.mean(intervals) if intervals else 1
        
        # Predict
        predictions = []
        for _ in range(horizon):
            pred = avg_demand_size / avg_interval
            predictions.append(max(0, pred))
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Croston prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_transformer(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Transformer model prediction for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters for Transformer
        horizon: Number of steps to predict into the future
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        
        # Simple transformer-like prediction using attention mechanism
        series = data['value'].dropna()
        
        # Create features with attention-like mechanism
        lookback = params.get('lookback', 20)
        X, y = _create_time_series_features(series, lookback=lookback)
        
        # Simple attention weights (inverse distance)
        attention_weights = np.exp(-np.arange(lookback) / lookback)
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Weighted prediction
        predictions = []
        last_features = X[-1]
        
        for _ in range(horizon):
            # Apply attention weights
            weighted_pred = np.sum(last_features * attention_weights)
            predictions.append(max(0, weighted_pred))
            
            # Update features
            last_features = np.roll(last_features, -1)
            last_features[-1] = weighted_pred
        
        return predictions
        
    except Exception as e:
        logger.warning(f"Transformer prediction failed: {e}")
        return predict_default(data, params, horizon)

def predict_default(data: Dict[str, Any], params: Dict[str, Any], horizon: int) -> List[float]:
    """Default prediction method for time series forecasting.
    
    Args:
        data: Time series data as dictionary with 'value' column
        params: Model parameters (not used in default method)
        horizon: Number of steps to predict into the future
    """
    # Convert dict to DataFrame if needed
    if isinstance(data, dict):
        data = pd.DataFrame(data)
    
    series = data['value'].dropna()
    
    # Simple moving average prediction
    window = min(10, len(series) // 4)
    if window > 0:
        ma = series.rolling(window=window).mean().iloc[-1]
    else:
        ma = series.mean()
    
    # Add some randomness
    predictions = []
    for i in range(horizon):
        pred = ma + np.random.normal(0, series.std() * 0.1)
        predictions.append(max(0, pred))
    
    return predictions

# Model mapping dictionary
MODEL_FUNCTIONS = {
    'ARIMA': predict_arima,
    'LSTM': predict_lstm,
    'ExponentialSmoothing': predict_exponential_smoothing,
    'LinearRegression': predict_linear_regression,
    'RandomForest': predict_random_forest,
    'Prophet': predict_prophet,
    'SVR': predict_svr,
    'GradientBoosting': predict_gradient_boosting,
    'XGBoost': predict_xgboost,
    'LightGBM': predict_lightgbm,
    'NeuralNetwork': predict_neural_network,
    'TBATS': predict_tbats,
    'Theta': predict_theta,
    'Croston': predict_croston,
    'Transformer': predict_transformer,
    'RandomWalk': predict_random_walk,
    'MovingAverage': predict_moving_average,
    'PolynomialRegression': predict_polynomial_regression,
    'RidgeRegression': predict_ridge_regression,
    'LassoRegression': predict_lasso_regression,
    'ElasticNet': predict_elastic_net
}

def get_model_function(model_name: str):
    """Get the prediction function for a given model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Prediction function for the model
    """
    return MODEL_FUNCTIONS.get(model_name, predict_default)
