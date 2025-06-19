import os
import json
import logging
import asyncio
import traceback
from datetime import datetime
import pytz
from flask import Flask, jsonify
import ccxt.async_support as ccxt
import talib
import numpy as np
import pandas as pd
import requests
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

def get_local_time():
    """Get current time in configured timezone"""
    try:
        utc_now = datetime.utcnow()
        local_tz = pytz.timezone(config.TIMEZONE)
        return utc_now.replace(tzinfo=pytz.UTC).astimezone(local_tz)
    except:
        return format_timestamp()

def format_timestamp(dt=None):
    """Format timestamp in local timezone"""
    if dt is None:
        dt = get_local_time()
    return dt.strftime(config.DATE_FORMAT)

async def get_top_futures():
    """Get top cryptocurrencies for futures trading"""
    try:
        exchange = ccxt.binance({
            'apiKey': os.environ.get('BINANCE_API_KEY'),
            'secret': os.environ.get('BINANCE_SECRET'),
            'sandbox': False,
            'options': {'defaultType': 'future'}
        })
        
        markets = await exchange.load_markets()
        futures_markets = [symbol for symbol in markets.keys() if 'USDT' in symbol and markets[symbol]['future']]
        
        # Get 24h volume for sorting
        tickers = await exchange.fetch_tickers()
        volume_data = [(symbol, tickers[symbol]['quoteVolume']) for symbol in futures_markets if symbol in tickers and tickers[symbol]['quoteVolume']]
        volume_data.sort(key=lambda x: x[1], reverse=True)
        
        await exchange.close()
        return [symbol for symbol, _ in volume_data[:config.CRYPTO_COUNT]]
    
    except Exception as e:
        logger.error(f"Error getting top futures: {e}")
        return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']

async def get_ohlcv_data(symbol, timeframe='1h', limit=100):
    """Get OHLCV data for technical analysis"""
    try:
        exchange = ccxt.binance({
            'apiKey': os.environ.get('BINANCE_API_KEY'),
            'secret': os.environ.get('BINANCE_SECRET'),
            'sandbox': False,
            'options': {'defaultType': 'future'}
        })
        
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        await exchange.close()
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    
    except Exception as e:
        logger.error(f"Error getting OHLCV for {symbol}: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate RSI, MACD, and Bollinger Bands"""
    try:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # RSI
        rsi = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        return {
            'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
            'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
            'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
            'macd_histogram': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
            'bb_upper': bb_upper[-1] if not np.isnan(bb_upper[-1]) else close[-1],
            'bb_middle': bb_middle[-1] if not np.isnan(bb_middle[-1]) else close[-1],
            'bb_lower': bb_lower[-1] if not np.isnan(bb_lower[-1]) else close[-1],
            'price': close[-1]
        }
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def generate_trading_signal(indicators):
    """Generate trading signals based on technical indicators"""
    try:
        if not indicators:
            return None
        
        rsi = indicators['rsi']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        price = indicators['price']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        bb_middle = indicators['bb_middle']
        
        signal_strength = 0
        signal_type = 'HOLD'
        reasons = []
        
        # RSI signals
        if rsi < 30:
            signal_strength += 30
            reasons.append(f"RSI oversold ({rsi:.1f})")
            signal_type = 'LONG'
        elif rsi > 70:
            signal_strength += 30
            reasons.append(f"RSI overbought ({rsi:.1f})")
            signal_type = 'SHORT'
        
        # MACD signals
        if macd > macd_signal and macd > 0:
            signal_strength += 25
            reasons.append("MACD bullish crossover")
            if signal_type != 'SHORT':
                signal_type = 'LONG'
        elif macd < macd_signal and macd < 0:
            signal_strength += 25
            reasons.append("MACD bearish crossover")
            if signal_type != 'LONG':
                signal_type = 'SHORT'
        
        # Bollinger Bands signals
        if price <= bb_lower:
            signal_strength += 20
            reasons.append("Price at lower BB")
            if signal_type != 'SHORT':
                signal_type = 'LONG'
        elif price >= bb_upper:
            signal_strength += 20
            reasons.append("Price at upper BB")
            if signal_type != 'LONG':
                signal_type = 'SHORT'
        
        # Additional confluence
        if len(reasons) >= 2:
            signal_strength += 15
        
        signal_strength = min(signal_strength, 100)
        
        # Only return signals with strength >= 60%
        if signal_strength >= 60:
            return {
                'type': signal_type,
                'strength': signal_strength,
                'rsi': rsi,
                'entry_price': price,
                'reasons': reasons
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return None

def calculate_targets_stops(entry_price, signal_type, strength):
    """Calculate target and stop loss levels"""
    try:
        risk_multiplier = strength / 100
        
        if signal_type == 'LONG':
            target1 = entry_price * (1 + 0.02 * risk_multiplier)
            target2 = entry_price * (1 + 0.04 * risk_multiplier)
            target3 = entry_price * (1 + 0.06 * risk_multiplier)
            stop_loss = entry_price * (1 - 0.03 * risk_multiplier)
        else:  # SHORT
            target1 = entry_price * (1 - 0.02 * risk_multiplier)
            target2 = entry_price * (1 - 0.04 * risk_multiplier)
            target3 = entry_price * (1 - 0.06 * risk_multiplier)
            stop_loss = entry_price * (1 + 0.03 * risk_multiplier)
        
        return {
            'target1': round(target1, 4),
            'target2': round(target2, 4),
            'target3': round(target3, 4),
            'stop_loss': round(stop_loss, 4)
        }
    
    except Exception as e:
        logger.error(f"Error calculating targets: {e}")
        return None

async def analyze_all_cryptos():
    """Analyze all cryptocurrencies and generate signals"""
    try:
        logger.info(f"Starting analysis at {format_timestamp()}")
        symbols = await get_top_futures()
        signals = []
        
        for symbol in symbols:
            try:
                df = await get_ohlcv_data(symbol)
                if df is not None and len(df) > 30:
                    indicators = calculate_technical_indicators(df)
                    signal = generate_trading_signal(indicators)
                    
                    if signal:
                        targets_stops = calculate_targets_stops(
                            signal['entry_price'], 
                            signal['type'], 
                            signal['strength']
                        )
                        
                        if targets_stops:
                            signal.update(targets_stops)
                            signal['symbol'] = symbol
                            signals.append(signal)
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by strength and get top signals
        signals.sort(key=lambda x: x['strength'], reverse=True)
        top_signals = signals[:config.SIGNAL_COUNT]
        
        logger.info(f"Analysis completed. Found {len(top_signals)} strong signals")
        return top_signals
    
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        return []

def send_telegram_message(message):
    """Send message to Telegram"""
    try:
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.error("Telegram credentials not found")
            return False
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Telegram error: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        return False

def format_signal_message(signals):
    """Format signals for Telegram message"""
    try:
        timestamp = format_timestamp()
        
        if not signals:
            return f"""🤖 <b>Crypto Futures Analysis</b>
📅 {timestamp}

👤 <b>{config.TELEGRAM_USER}</b>

📊 <b>Market Summary:</b>
- Analyzed: {config.CRYPTO_COUNT} top cryptocurrencies
- Strong Signals: 0
- Average Strength: 0%

💡 <i>No strong signals found at this time. Next update in 2 hours</i>"""
        
        total_strength = sum(s['strength'] for s in signals)
        avg_strength = total_strength / len(signals)
        
        message = f"""🎯 <b>Crypto Futures Analysis</b>
📅 {timestamp}

👤 <b>{config.TELEGRAM_USER}</b>

📊 <b>Market Summary:</b>
- Analyzed: {config.CRYPTO_COUNT} top cryptocurrencies
- Strong Signals: {len(signals)}
- Average Strength: {avg_strength:.0f}%

🏆 <b>Top Trading Signals:</b>

"""
        
        for i, signal in enumerate(signals, 1):
            emoji = "🔴" if signal['type'] == 'SHORT' else "🟢"
            symbol_clean = signal['symbol'].replace('/USDT', '')
            
            message += f"""<b>{i}.</b> {emoji} <b>{symbol_clean}/USDT {signal['type']}</b>
💎 <b>Strength:</b> {signal['strength']}% | <b>RSI:</b> {signal['rsi']:.1f}
💰 <b>Entry:</b> ${signal['entry_price']:.4f} - ${signal['target1']:.4f}
🎯 <b>Targets:</b> ${signal['target1']:.4f} | ${signal['target2']:.4f} | ${signal['target3']:.4f}
🛑 <b>Stop Loss:</b> ${signal['stop_loss']:.4f}

"""
        
        message += f"⚡ <i>Real-time analysis of top {config.CRYPTO_COUNT} cryptos | Next update in 2 hours</i>"
        
        return message
    
    except Exception as e:
        logger.error(f"Error formatting message: {e}")
        return f"Error formatting analysis results at {format_timestamp()}"

# Flask routes
@app.route('/')
def home():
    return jsonify({
        'service': 'crypto-futures-bot',
        'status': 'running',
        'version': 'real-2.0',
        'timestamp': format_timestamp(),
        'timezone': config.TIMEZONE,
        'features': {
            'real_data': True,
            'technical_analysis': True,
            'telegram_integration': True,
            'automated_signals': True
        },
        'config': {
            'crypto_count': config.CRYPTO_COUNT,
            'signal_count': config.SIGNAL_COUNT,
            'telegram_user': config.TELEGRAM_USER
        },
        'analysis_type': 'real-time'
    })

@app.route('/status')
def status():
    return jsonify({
        'service': 'crypto-futures-bot',
        'status': 'running',
        'version': 'real-2.0',
        'timestamp': format_timestamp(),
        'timezone': config.TIMEZONE,
        'features': {
            'real_data': True,
            'technical_analysis': True,
            'telegram_integration': True,
            'automated_signals': True
        },
        'config': {
            'crypto_count': config.CRYPTO_COUNT,
            'signal_count': config.SIGNAL_COUNT,
            'telegram_user': config.TELEGRAM_USER
        },
        'analysis_type': 'real-time'
    })

@app.route('/analyze')
def analyze():
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        signals = loop.run_until_complete(analyze_all_cryptos())
        loop.close()
        
        message = format_signal_message(signals)
        telegram_sent = send_telegram_message(message)
        
        return jsonify({
            'success': True,
            'timestamp': format_timestamp(),
            'timezone': config.TIMEZONE,
            'signals_found': len(signals),
            'telegram_sent': telegram_sent,
            'signals': signals
        })
    
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': format_timestamp(),
            'timezone': config.TIMEZONE
        })

@app.route('/test')
def test():
    try:
        test_message = f"""🧪 <b>Test Message</b>
📅 {format_timestamp()}
🌍 <b>Timezone:</b> {config.TIMEZONE}

✅ Bot is working correctly!
👤 <b>User:</b> {config.TELEGRAM_USER}"""
        
        telegram_sent = send_telegram_message(test_message)
        
        return jsonify({
            'success': True,
            'message': 'Test completed',
            'timestamp': format_timestamp(),
            'timezone': config.TIMEZONE,
            'telegram_sent': telegram_sent
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': format_timestamp(),
            'timezone': config.TIMEZONE
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
# Timezone ready - 2025-06-19
