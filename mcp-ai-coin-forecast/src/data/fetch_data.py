import requests
import logging
from datetime import datetime
from typing import List, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CoinGeckoAPIError(Exception):
    """Custom exception for CoinGecko API errors"""
    pass

def validate_coin(coin_id: str) -> bool:
    """
    Validate if a coin exists on CoinGecko
    """
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def get_historical_market_cap(coin_id: str = 'bitcoin', 
                            vs_currency: str = 'usd', 
                            days: int = 30,
                            retries: int = 3) -> List[Tuple[str, float]]:
    """
    Fetch historical market cap data for a coin from CoinGecko.
    
    Args:
        coin_id: CoinGecko coin id (e.g., 'bitcoin')
        vs_currency: Currency to compare against (e.g., 'usd')
        days: Number of days in the past to fetch
        retries: Number of retry attempts for API calls
        
    Returns:
        List of (date, market_cap) tuples
        
    Raises:
        CoinGeckoAPIError: If API request fails after all retries
    """
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'daily'
    }
    
    for attempt in range(retries):
        try:
            logger.info(f"Fetching data for {coin_id} (Attempt {attempt + 1}/{retries})")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 429:  # Rate limit
                wait_time = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            market_caps = data.get('market_caps', [])
            
            if not market_caps:
                logger.warning(f"No market cap data found for {coin_id}")
                return []
            
            # Convert timestamps to date strings
            result = []
            for ts, cap in market_caps:
                date = datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d')
                result.append((date, cap))
            
            logger.info(f"Successfully fetched {len(result)} days of data for {coin_id}")
            return result
            
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                logger.error(f"Failed to fetch data for {coin_id}: {str(e)}")
                raise CoinGeckoAPIError(f"Failed to fetch data after {retries} attempts: {str(e)}")
            logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
            
    return []