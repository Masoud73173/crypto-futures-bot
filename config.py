# Configuration file for Crypto Futures Analysis Bot
import os

# Output Settings
OUTPUT_TYPE = "MULTI"
OUTPUT_SETTINGS = {
    'html': True,
    'telegram': True, 
    'whatsapp': False
}

# Bot Information
BOT_INFO = {
    'name': 'Crypto_Futures_Analysis_Bot',
    'token': '7754175422:AAFOHaxVwphnfm43I_Y7BoVdSHmXKcgdQQA',
    'username': '@Crypto_Futures_Analysis_Bot'
}

# User Information
USER_INFO = {
    'first_name': 'Masoud',
    'last_name': 'Haddad',
    'username': '@MasoudHaddad69',
    'chat_id': '55174977',
    'language': 'en'
}

# Telegram Settings
TELEGRAM_BOT_TOKEN = "7754175422:AAFOHaxVwphnfm43I_Y7BoVdSHmXKcgdQQA"
TELEGRAM_CHAT_ID = "55174977"

# Analysis Settings
REPORT_INTERVAL_HOURS = 2  # Every 2 hours
CRYPTO_COUNT = 200  # Analyze 200 cryptocurrencies
SIGNAL_COUNT = 10   # Generate 10 trading signals
TIMEFRAME = '1h'    # 1 hour timeframe for analysis

# Technical Analysis Settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Trading Signal Thresholds
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MIN_VOLUME_24H = 1000000  # Minimum 24h volume in USD
MIN_PRICE_CHANGE = 0.5    # Minimum price change percentage

# Risk Management
MAX_RISK_PER_TRADE = 3    # Maximum 3% risk per trade
STOP_LOSS_PERCENTAGE = 2  # 2% stop loss
TAKE_PROFIT_LEVELS = [3, 6, 12]  # Take profit levels in percentage

# Exchange Settings
EXCHANGE = 'binance'
MARKET_TYPE = 'futures'

# File Paths
HTML_OUTPUT_PATH = 'crypto_analysis_report.html'
LOG_FILE_PATH = 'crypto_bot.log'

# Time Settings
TIMEZONE = 'UTC'
DATE_FORMAT = '%Y/%m/%d - %H:%M'

# HTML Template Settings
HTML_TITLE = "Crypto Futures Analysis Report"
REPORT_STYLE = "modern"  # modern, classic, minimal

# Error Handling
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Logging Level
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

print("✅ Configuration loaded successfully!")
print(f"📊 Analysis: {CRYPTO_COUNT} cryptos every {REPORT_INTERVAL_HOURS} hours")
print(f"📱 Telegram: {USER_INFO['username']} ({USER_INFO['chat_id']})")
print(f"🎯 Signals: {SIGNAL_COUNT} trading signals per report")
