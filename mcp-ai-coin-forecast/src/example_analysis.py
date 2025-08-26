import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from analysis.multi_crypto_analyzer import MultiCryptoAnalyzer
from models.forecast_model import ForecastModel
from utils.backtesting import Backtester
from data.fetch_data import get_historical_market_cap, validate_coin
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_backtest_results(timeframe, df, results, backtest_results):
    """Process and display backtest results for a timeframe"""
    # Print current and predicted values
    current_value = df['market_cap'].iloc[-1]
    predicted_value = results[timeframe]['prediction']
    if current_value == 0 or pd.isna(current_value):
        predicted_change = float('nan')
    else:
        predicted_change = ((predicted_value - current_value) / current_value * 100)

    print(f"\nTimeframe: {timeframe}")
    print(f"Current market cap: ${current_value:,.2f}")
    print(f"Predicted next value: ${predicted_value:,.2f}")
    print(f"Predicted change: {predicted_change:,.2f}%")

    if backtest_results and 'metrics' in backtest_results:
        print("\nBacktesting Metrics:")
        metrics = backtest_results['metrics']
        print(f"MSE: {metrics.get('mse', 0):.2f}")
        print(f"RMSE: {metrics.get('rmse', 0):.2f}")
        print(f"MAPE: {metrics.get('mape', 0):.2f}%")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")

def analyze_multiple_coins():
    """Analyze multiple cryptocurrencies in parallel"""
    # List of coins to analyze
    coins = ['bitcoin', 'ethereum', 'cardano', 'solana', 'polkadot', 'pi-network']
    
    logger.info(f"Starting analysis for {len(coins)} coins...")
    
    # Initialize analyzer
    analyzer = MultiCryptoAnalyzer(
        coins=coins,
        timeframe='daily',
        model_type='rf',
        max_workers=4
    )
    
    # Run analysis
    results = analyzer.run_analysis()
    
    # Generate and display summary report
    summary_df = analyzer.generate_summary_report()
    print("\nAnalysis Summary:")
    print(summary_df)
    
    # Plot comparative results
    analyzer.plot_comparative_results()
    
    return results

def analyze_single_coin_multiple_timeframes(coin_id='bitcoin'):
    """
    Analyze a single coin across different timeframes.
    
    Args:
        coin_id (str): CoinGecko coin identifier
        
    Returns:
        dict: Analysis results for each timeframe or None if validation fails
    """
    # Validate coin first
    if not validate_coin(coin_id):
        print(f"\nError: '{coin_id}' is not a valid coin on CoinGecko.")
        print("Please check the coin ID and try again.")
        return None
        
    timeframes = ['1h', '4h', 'daily', 'weekly']
    results = {}
    
    try:
        for timeframe in timeframes:
            logger.info(f"Analyzing {coin_id} with {timeframe} timeframe...")
            
            try:
                # Create and train model
                model = ForecastModel(
                    model_type='rf',
                    timeframe=timeframe
                )
                
                # Get data
                raw_data = get_historical_market_cap(coin_id=coin_id, days=120)
                if not raw_data:
                    logger.warning(f"No data received for {timeframe}")
                    continue
                    
                df = pd.DataFrame(raw_data, columns=['date', 'market_cap'])
                
                # Ensure we have enough data
                if len(df) < 30:  # Minimum required data points
                    logger.warning(f"Insufficient data for {timeframe} analysis")
                    continue
                
                # Train and backtest
                X_test, y_test = model.train(df)
                backtester = Backtester(model, df)
                backtest_results = backtester.run()
                
                # Store results
                results[timeframe] = {
                    'model': model,
                    'backtest_results': backtest_results,
                    'current_value': df['market_cap'].iloc[-1],
                    'prediction': model.predict_future(df)
                }
                
                # Process and display results
                process_backtest_results(timeframe, df, results, backtest_results)
                
                # Plot backtest results
                try:
                    backtester.plot_results()
                except Exception as e:
                    logger.warning(f"Could not plot results: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error analyzing {timeframe}: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Fatal error during analysis: {str(e)}")
        raise

def main():
    """Main function to demonstrate various analyses"""
    print("=== Cryptocurrency Market Cap Forecast Analysis ===")
    
    while True:
        print("\nOptions:")
        print("1. Analyze multiple cryptocurrencies")
        print("2. Analyze single cryptocurrency with multiple timeframes")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            try:
                results = analyze_multiple_coins()
            except Exception as e:
                print(f"\nError during analysis: {str(e)}")
                continue
            
        elif choice == '2':
            coin_id = input("Enter coin ID (e.g., bitcoin, ethereum, pi-network): ").strip().lower()
            try:
                results = analyze_single_coin_multiple_timeframes(coin_id)
                if results is None:  # Coin validation failed
                    continue
            except Exception as e:
                print(f"\nError during analysis: {str(e)}")
                continue
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\nA fatal error occurred: {str(e)}")
