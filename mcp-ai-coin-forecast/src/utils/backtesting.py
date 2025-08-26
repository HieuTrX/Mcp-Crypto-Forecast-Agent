import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, model, data: pd.DataFrame, window_size: int = 30, stride: int = 7):
        """
        Initialize backtester.
        
        Args:
            model: Trained forecast model instance
            data: Historical data DataFrame
            window_size: Size of training window in days
            stride: Number of days to move forward in each iteration
        """
        self.model = model
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.results = []
        self.metrics = {}

    def run(self) -> Dict:
        """
        Run backtesting simulation.
        
        Returns:
            Dictionary containing backtesting results and metrics
        """
        try:
            logger.info("Starting backtesting simulation...")
            
            # We'll use a rolling window approach
            window_size = min(30, len(self.data) - 2)  # Minimum 2 points needed for prediction
            self.results = []
            
            if len(self.data) < window_size + 2:
                logger.warning("Not enough data for backtesting")
                return {
                    'metrics': {
                        'mse': 0,
                        'rmse': 0,
                        'mape': 0,
                        'win_rate': 0
                    },
                    'results': []
                }
            
            for i in range(len(self.data) - window_size - 1):
                # Get the training window
                train_data = self.data.iloc[i:i+window_size].copy()
                try:
                    # Train model on this window
                    self.model.train(train_data)
                    # Make prediction
                    prediction = self.model.predict_future(train_data)
                    # Get actual next value
                    actual = self.data.iloc[i+window_size+1]['market_cap']
                    # Calculate error
                    error = abs(prediction - actual)
                    if actual == 0 or pd.isna(actual):
                        error_pct = float('nan')
                    else:
                        error_pct = (error / actual) * 100
                    # Store results
                    self.results.append({
                        'date': self.data.iloc[i+window_size+1]['date'],
                        'predicted_value': prediction,
                        'actual_value': actual,
                        'error': error,
                        'error_pct': error_pct
                    })
                except Exception as e:
                    logger.warning(f"Error in window {i}: {str(e)}")
                    continue
            # Calculate overall metrics
            self._calculate_metrics()
            logger.info("Backtesting completed successfully")
            return self.get_results()
        except Exception as e:
            logger.error(f"Backtesting failed: {str(e)}")
            return {
                'metrics': {
                    'mse': 0,
                    'rmse': 0,
                    'mape': 0,
                    'win_rate': 0
                },
                'results': []
            }

    def _calculate_metrics(self):
        """Calculate performance metrics for backtesting results"""
        if not self.results:
            self.metrics = {
                'mse': 0,
                'rmse': 0,
                'mape': 0,
                'avg_error_pct': 0,
                'max_error_pct': 0,
                'min_error_pct': 0,
                'win_rate': 0
            }
            return

        df_results = pd.DataFrame(self.results)
        
        self.metrics = {
            'mse': mean_squared_error(df_results['actual_value'], df_results['predicted_value']),
            'rmse': np.sqrt(mean_squared_error(df_results['actual_value'], df_results['predicted_value'])),
            'mape': mean_absolute_percentage_error(df_results['actual_value'], df_results['predicted_value']) * 100,
            'avg_error_pct': df_results['error_pct'].mean(),
            'max_error_pct': df_results['error_pct'].max(),
            'min_error_pct': df_results['error_pct'].min(),
            'win_rate': (df_results['error_pct'] < 10).mean() * 100  # % of predictions with <10% error
        }

    def plot_results(self):
        """Plot backtesting results"""
        if not self.results:
            logger.warning("No results to plot")
            return
            
        df_results = pd.DataFrame(self.results)
        
        plt.figure(figsize=(15, 10))
        
        # Plot predictions vs actual values
        plt.subplot(2, 1, 1)
        plt.plot(df_results['date'], df_results['actual_value'], label='Actual', color='blue')
        plt.plot(df_results['date'], df_results['predicted_value'], label='Predicted', color='red', linestyle='--')
        plt.title('Backtesting Results: Predicted vs Actual Values')
        plt.xlabel('Date')
        plt.ylabel('Market Cap')
        plt.legend()
        
        # Plot prediction errors
        plt.subplot(2, 1, 2)
        plt.plot(df_results['date'], df_results['error_pct'], color='green')
        plt.title('Prediction Error Percentage')
        plt.xlabel('Date')
        plt.ylabel('Error %')
        plt.axhline(y=10, color='r', linestyle='--', label='10% Error Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def get_results(self) -> Dict:
        """Get backtesting results and metrics"""
        try:
            if not self.results:
                return {
                    'metrics': {
                        'mse': 0,
                        'rmse': 0,
                        'mape': 0,
                        'win_rate': 0
                    },
                    'results': []
                }
            
            self._calculate_metrics()
            return {
                'results': self.results,
                'metrics': self.metrics
            }
        except Exception as e:
            logger.error(f"Error getting results: {str(e)}")
            return {
                'metrics': {
                    'mse': 0,
                    'rmse': 0,
                    'mape': 0,
                    'win_rate': 0
                },
                'results': []
            }

    def generate_report(self) -> str:
        """Generate a detailed backtesting report"""
        report = []
        report.append("=== Backtesting Report ===")
        report.append(f"Time Period: {self.results[0]['date']} to {self.results[-1]['date']}")
        report.append(f"Total Predictions: {len(self.results)}")
        report.append("\nPerformance Metrics:")
        
        for metric, value in self.metrics.items():
            report.append(f"{metric.upper()}: {value:.2f}")
            
        report.append(f"\nPrediction Win Rate (Error < 10%): {self.metrics['win_rate']:.2f}%")
        
        return "\n".join(report)
