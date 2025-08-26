import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import Dict, Tuple, List, Optional
import ta  # Technical Analysis library for additional indicators

logger = logging.getLogger(__name__)

class ForecastModel:
    def predict_future(self, df):
        """
        Predict the next market cap value using the most recent data.
        Args:
            df (pd.DataFrame): DataFrame with at least as many rows as the lookback window
        Returns:
            float: Predicted next market cap value
        """
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        # Create features for the latest window
        df_processed = self._create_features(df)
        X, _ = self._prepare_data(df_processed)
        # Use the last available feature row for prediction
        X_last = X[-1].reshape(1, -1)
        X_last_scaled = self.scaler.transform(X_last)
        pred = self.model.predict(X_last_scaled)
        return float(pred[0])
    def __init__(self, model_type='rf', forecast_days=7, timeframe='daily'):
        """
        Initialize the forecast model.
        
        Args:
            model_type: Type of model ('linear', 'rf', 'ridge', 'lasso', 'svr')
            forecast_days: Number of days to forecast into the future
            timeframe: Data timeframe ('daily', 'hourly', '4h', '1h')
        """
        self.model_type = model_type
        self.pipeline = None
        self.forecast_days = forecast_days
        self.lookback_window = 30
        self.feature_importance = None
        self.metrics = {}
        self.timeframe = timeframe
        self.best_params = None
        
        # Define model parameters for grid search
        self.param_grid = {
            'rf': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5, 10]
            },
            'ridge': {
                'model__alpha': [0.1, 1.0, 10.0]
            },
            'lasso': {
                'model__alpha': [0.1, 1.0, 10.0]
            },
            'svr': {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['rbf', 'linear']
            }
        }

    def _create_features(self, df):
        """Create features for time series forecasting"""
        df = df.copy()
        # Technical indicators
        df['MA7'] = df['market_cap'].rolling(window=7).mean()
        df['MA30'] = df['market_cap'].rolling(window=30).mean()
        df['std_dev'] = df['market_cap'].rolling(window=7).std()
        # Momentum
        df['momentum'] = df['market_cap'].pct_change(periods=7)
        df['roc'] = df['market_cap'].pct_change(periods=14)
        # Fill NaN values
        df = df.bfill()
        return df

    def _prepare_data(self, df):
        """Prepare features and target for training"""
        # Create features for training
        features = ['market_cap', 'MA7', 'MA30', 'std_dev', 'momentum', 'roc']
        X = df[features].values
        
        # Create target (next day's market cap)
        y = df['market_cap'].shift(-1).values
        
        # Remove the last row where we don't have a target
        X = X[:-1]
        y = y[:-1]
        
        return X, y

    def train(self, df):
        """Train the model using historical market cap data"""
        logger.info("Starting model training...")
        
        # Create features
        df_processed = self._create_features(df)
        X, y = self._prepare_data(df_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model based on type
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        elif self.model_type == 'svr':
            self.model = SVR(kernel='rbf', C=1.0)
        else:
            raise ValueError("Unsupported model type. Choose 'rf', 'linear', 'ridge', 'lasso', or 'svr'")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        self.metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }
        
        # Store feature importance for RF model
        if self.model_type == 'rf':
            features = ['market_cap', 'MA7', 'MA30', 'std_dev', 'momentum', 'roc']
            self.feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        logger.info(f"Model training completed. MAPE: {self.metrics['mape']:.2f}%")
        return X_test_scaled, y_test

    def predict(self, X_test):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        return self.model.predict(X_test)