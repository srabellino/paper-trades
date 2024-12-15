import yfinance as yf
from datetime import datetime, timedelta, date, timezone
import numpy as np
from scipy.stats import norm
import requests
import logging
import boto3
import uuid

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Alpaca API credentials
API_KEY = 'PKL4KJP2396KTLE88NBQ'
SECRET_KEY = 'KLKJtoBaOwdiZ3elhqKRdafiLD1sFFlpDcI5IpQB'
PAPER = True 

# AWS DynamoDB setup
REGION_NAME = 'us-east-2' 
dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
table = dynamodb.Table('OptionsTransactions')

# symbols to trade
SYMBOLS = ['QQQ', 'SPY', 'IWM', 'HYG', 'SLV', 'GLD', 'ARKK', 'NVDA',
            'AAPL', 'MSFT', 'TSLA', 'CMG', 'COST', 'LLY', 'NFLX',
            'GOOGL', 'NUE', 'AMZN', 'NVO', 'MSTR', 'LVMUY']


class OptionTradingBot:
    def __init__(self, api_key, secret_key, paper=True):
        """
        Initialize the trading bot with Alpaca API credentials.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.trade_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
    
    def calc_delta(self, contracts, last_close_price):
        """
        Calculate the delta for each option contract.
        """
        r = 0.05  # Risk-free interest rate
        sigma = 0.20  # Volatility
        contract_deltas = []

        for contract in contracts:
            S = last_close_price
            K = contract.strike_price
            T = (contract.expiration_date - contract.close_price_date).days / 365
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            delta = norm.cdf(d1) - 1  # Delta for put option
            contract_deltas.append(delta)
        
        logging.debug(f"Calculated deltas: {contract_deltas}")
        return contract_deltas

    def find_target_delta_contract(self, snapshots, target_delta=0.3):
        """
        Find the option contract with the closest delta to the target.
        """
        closest_contract = None
        last_delta_dif = float('inf')

        for key, snapshot in snapshots.items():
            try:
                delta_diff = abs(abs(snapshot['greeks']['delta']) - target_delta)
            except Exception as e:
                logging.warning(f'No greeks for contract {key}')
                continue
            if delta_diff < last_delta_dif:
                last_delta_dif = delta_diff
                closest_contract = snapshot
                closest_contract['symbol'] = key
        
        if last_delta_dif < 0.1:
            logging.info(f"Found contract with target delta: {closest_contract['symbol']}")
            return closest_contract
        else:
            logging.info(f"No contract found close to target delta {target_delta}")
            return None

    def get_snapshots(self, symbol, params):
        """
        Fetch the snapshot data for options contracts from Alpaca API.
        """
        url = f'https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}'
        headers = {
            'accept': 'application/json',
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            logging.info(f"Fetched snapshots for {symbol}")
            return response.json()['snapshots']
        else:
            logging.error(f"Failed to fetch snapshots for {symbol}. Error: {response.text}")
            return None

    def is_buy_signal(self, stock_data, fast_column, slow_column):
        """
        Determine if there is a buy signal based on moving averages.
        """
        if stock_data.empty:
            raise ValueError('DataFrame is empty. Cannot check buy signal.')

        most_recent_date = stock_data.index[-1]
        fast_value = stock_data.loc[most_recent_date, fast_column]
        slow_value = stock_data.loc[most_recent_date, slow_column]
        return fast_value > slow_value

    def process_stock_data(self, symbol, start_date, end_date):
        """
        Fetch stock data and determine if a buy or sell signal is generated.
        """
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            logging.error(f"Failed to fetch data for {symbol}")
            return

        # Calculate moving averages and MACD
        stock_df = stock_data['Close'].to_frame()
        stock_df['Moving_Avg_1'] = stock_df['Close'].rolling(window=3).mean()
        stock_df['Moving_Avg_2'] = stock_df['Close'].rolling(window=6).mean()

        stock_df['EMA_Short'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
        stock_df['EMA_Long'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
        stock_df['MACD'] = stock_df['EMA_Short'] - stock_df['EMA_Long']
        stock_df['Signal_Line'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()

        stock_df['EMA_Short_2'] = stock_df['Close'].ewm(span=3, adjust=False).mean()
        stock_df['EMA_Long_2'] = stock_df['Close'].ewm(span=9, adjust=False).mean()
        stock_df['MACD_2'] = stock_df['EMA_Short_2'] - stock_df['EMA_Long_2']
        stock_df['Signal_Line_2'] = stock_df['MACD_2'].ewm(span=5, adjust=False).mean()

        # Determine buy/sell signals
        moving_avg_signal = self.is_buy_signal(stock_df, 'Moving_Avg_1', 'Moving_Avg_2')
        macd_signal_1 = self.is_buy_signal(stock_df, 'MACD', 'Signal_Line')
        macd_signal_2 = self.is_buy_signal(stock_df, 'MACD_2', 'Signal_Line_2')

        if not moving_avg_signal and not macd_signal_1 and not macd_signal_2:
            self.process_option_trading(symbol, stock_df, 'put')
        elif moving_avg_signal and macd_signal_1 and macd_signal_2:
            self.process_option_trading(symbol, stock_df, 'call')
        logging.info(f"Completed processing for {symbol}")

    def get_mid_price(self, snapshot):
        """
        Get the mid price for an option contract from the snapshot data.
        """
        bid = snapshot['latestQuote']['bp']
        ask = snapshot['latestQuote']['ap']
        mid = round(((ask + bid) / 2), 2)
        logging.debug(f"Mid price for contract {snapshot['symbol']}: {mid}")
        return mid

    def log_transaction(self, ticker, symbol, option_type, qty, price):
        """
        Log a transaction to DynamoDB.
        """
        transaction_id = str(uuid.uuid4())  # Generate a unique ID for the transaction
        timestamp = datetime.now(timezone.utc).isoformat()  # Get the current timestamp
        date = datetime.today().strftime('%Y-%m-%d')
        try:
            table.put_item(
                Item={
                    'transaction_id': transaction_id,
                    'symbol': symbol,
                    'ticker': ticker,
                    'option_type': option_type,
                    'quantity': qty,
                    'price': str(price),
                    'cost_basis': str(price*qty*100),
                    'timestamp': timestamp,
                    'date': date
                }
            )
            logging.info(f"Logged transaction to DynamoDB: {transaction_id}, {symbol}, {option_type}, {qty}, {price}S")
        except Exception as e:
            logging.error(f"Failed to log transaction: {e}")

    def process_option_trading(self, symbol, stock_df, option_type):
        """
        Process option trading based on the signals received.
        """
        last_close = stock_df.loc[stock_df.index[-1], 'Close']
        exp_date_gte = date.today() + timedelta(days=25)
        exp_date_lte = date.today() + timedelta(days=35)

        params = {
            'feed': 'indicative',
            'limit': 100,
            'type': option_type,
            'strike_price_gte': str(last_close * .75),
            'strike_price_lte': str(last_close * 1.25),
            'expiration_date_gte': exp_date_gte,
            'expiration_date_lte': exp_date_lte
        }
        snapshots = self.get_snapshots(symbol, params)
        if snapshots:
            target_contract = self.find_target_delta_contract(snapshots)
            if target_contract:
                mid_price = self.get_mid_price(target_contract)
                order_request = LimitOrderRequest(
                    symbol=target_contract['symbol'],
                    qty=1,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    limit_price=mid_price,
                    time_in_force=TimeInForce.DAY
                )
                self.trade_client.submit_order(order_request)
                logging.info(f"Placed {option_type} order for {symbol}")
                # Log the transaction
                self.log_transaction(symbol, target_contract['symbol'], option_type, 1, mid_price)
            else:
                logging.warning(f"No suitable contract found for {symbol}")
        else:
            logging.error(f"No snapshots found for {symbol}")

# Initialize and run the bot in AWS Lambda
def lambda_handler(event, context):
    """
    Lambda entry point: Process a list of symbols and trade options.
    """
    end_date = datetime.today()
    start_date = end_date - timedelta(days=200)
    bot = OptionTradingBot(api_key=API_KEY, secret_key=SECRET_KEY, paper=PAPER)
    for symbol in SYMBOLS:
        bot.process_stock_data(symbol, start_date, end_date)
    return

# Local testing
# lambda_handler('', '')
