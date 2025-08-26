import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import logging
from datetime import datetime
from data.fetch_data import get_historical_market_cap
from models.forecast_model import ForecastModel
from utils.backtesting import Backtester
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MultiCryptoAnalyzer:
    def __init__(self, coins: List[str], timeframe: str = 'daily', 
                 model_type: str = 'rf', max_workers: int = 4):
        """
        Initialize multi-cryptocurrency analyzer.
        
        Args:
            coins: List of CoinGecko coin IDs
            timeframe: Data timeframe ('1h', '4h', 'daily', 'weekly')
            model_type: Type of model to use
            max_workers: Maximum number of parallel workers
        """
        self.coins = coins
        self.timeframe = timeframe
        self.model_type = model_type
        self.max_workers = max_workers
        self.results = {}

    def analyze_coin(self, coin_id: str) -> Dict[str, Any]:
        """Analyze a single cryptocurrency"""
        try:
            # Fetch data
            raw_data = get_historical_market_cap(
                coin_id=coin_id,
                vs_currency='usd',
                days=365 if self.timeframe == 'daily' else 30
            )
            
            if not raw_data:
                logger.error(f"No data received for {coin_id}")
                return None
                
            # Create DataFrame
            df = pd.DataFrame(raw_data, columns=['date', 'market_cap'])
            df['date'] = pd.to_datetime(df['date'])
            
            # Initialize and train model
            model = ForecastModel(
                model_type=self.model_type,
                timeframe=self.timeframe
            )
            
            # Train model and get predictions
            X_test, y_test = model.train(df)
            future_pred = model.predict_future(df)
            
            # Run backtesting
            backtester = Backtester(model, df)
            backtest_results = backtester.run()
            
            return {
                'coin_id': coin_id,
                'model': model,
                'predictions': future_pred,
                'current_value': df['market_cap'].iloc[-1],
                'backtest_results': backtest_results,
                'feature_importance': model.feature_importance if model.model_type == 'rf' else None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {coin_id}: {str(e)}")
            return None

    def run_analysis(self) -> Dict[str, Dict]:
        """
        Run parallel analysis for all cryptocurrencies.
        
        Returns:
            Dictionary containing analysis results for each coin
        """
        logger.info(f"Starting analysis for {len(self.coins)} coins...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_coin = {
                executor.submit(self.analyze_coin, coin): coin 
                for coin in self.coins
            }
            
            with tqdm(total=len(self.coins), desc="Analyzing coins") as pbar:
                for future in as_completed(future_to_coin):
                    coin = future_to_coin[future]
                    try:
                        result = future.result()
                        if result:
                            self.results[coin] = result
                    except Exception as e:
                        logger.error(f"Analysis failed for {coin}: {str(e)}")
                    pbar.update(1)
        
        return self.results

    def generate_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of all analyzed coins"""
        summary_data = []
        
        for coin_id, result in self.results.items():
            if not result:
                continue
                
            metrics = result['backtest_results']['metrics']
            if result['current_value'] == 0 or pd.isna(result['current_value']):
                pred_change = float('nan')
            else:
                pred_change = ((result['predictions'] - result['current_value']) / result['current_value'] * 100)
            
            summary_data.append({
                'coin_id': coin_id,
                'current_value': result['current_value'],
                'predicted_change_%': pred_change,
                'backtest_mape': metrics['mape'],
                'backtest_win_rate': metrics['win_rate'],
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(summary_data)

    def plot_comparative_results(self):
        """Plot comparative analysis results"""
        if not self.results:
            logger.warning("No results to plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot predicted changes
        plt.subplot(2, 1, 1)
        changes = []
        coins = []
        
        for coin_id, result in self.results.items():
            if not result:
                continue
            if result['current_value'] == 0 or pd.isna(result['current_value']):
                pred_change = float('nan')
            else:
                pred_change = ((result['predictions'] - result['current_value']) / result['current_value'] * 100)
            changes.append(pred_change)
            coins.append(coin_id)
        
        plt.bar(coins, changes)
        plt.title('Predicted Price Changes (%)')
        plt.xticks(rotation=45)
        
        # Plot backtest win rates
        plt.subplot(2, 1, 2)
        win_rates = []
        
        for coin_id, result in self.results.items():
            if not result:
                continue
            win_rates.append(result['backtest_results']['metrics']['win_rate'])
        
        plt.bar(coins, win_rates)
        plt.title('Backtest Win Rates (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
