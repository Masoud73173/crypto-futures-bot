#!/usr/bin/env python3
"""
Crypto Futures Analysis Bot - Real Analysis Version
"""

from flask import Flask, jsonify
import requests
import json
import logging
from datetime import datetime
import os
import ccxt
import pandas as pd
import numpy as np
import ta
import time
from typing import List, Dict, Optional
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class RealCryptoAnalyzer:
    """Real cryptocurrency analysis with live data"""
    
    def __init__(self):
        """Initialize with exchange"""
        self.exchange = ccxt.binance({
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
    def get_top_symbols(self, limit=200):
        """Get top crypto symbols by volume"""
        try:
            logger.info(f"🔍 Fetching top {limit} crypto symbols...")
            
            # Get 24h ticker data
            tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs and sort by volume
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol and ticker['quoteVolume']:
                    usdt_pairs.append({
                        'symbol': symbol,
                        'volume': ticker['quoteVolume'],
                        'price': ticker['last'],
                        'change': ticker['percentage']
                    })
            
            # Sort by volume and take top N
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            top_symbols = usdt_pairs[:limit]
            
            logger.info(f"✅ Found {len(top_symbols)} top symbols")
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ Error fetching symbols: {str(e)}")
            return []
    
    def get_ohlcv_data(self, symbol, timeframe='1h', limit=100):
        """Get OHLCV data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Error fetching OHLCV for {symbol}: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {str(e)}")
            return df
    
    def generate_signal(self, symbol_data, df):
        """Generate trading signal"""
        try:
            if len(df) < 30:
                return None
                
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            price = latest['close']
            rsi = latest['rsi']
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            
            signal_type = None
            strength = 0
            reasons = []
            
            # LONG Signal Conditions
            long_score = 0
            if rsi < 40 and rsi > 25:
                long_score += 25
                reasons.append("RSI oversold")
            
            if macd > macd_signal and prev['macd'] <= prev['macd_signal']:
                long_score += 30
                reasons.append("MACD bullish crossover")
            
            if latest['ema_12'] > latest['ema_26']:
                long_score += 20
                reasons.append("EMA alignment bullish")
            
            if price <= latest['bb_lower'] * 1.02:
                long_score += 25
                reasons.append("Near Bollinger lower band")
            
            # SHORT Signal Conditions  
            short_score = 0
            if rsi > 60 and rsi < 75:
                short_score += 25
                reasons.append("RSI overbought")
            
            if macd < macd_signal and prev['macd'] >= prev['macd_signal']:
                short_score += 30
                reasons.append("MACD bearish crossover")
            
            if latest['ema_12'] < latest['ema_26']:
                short_score += 20
                reasons.append("EMA alignment bearish")
            
            if price >= latest['bb_upper'] * 0.98:
                short_score += 25
                reasons.append("Near Bollinger upper band")
            
            # Determine signal
            if long_score >= 50:
                signal_type = "LONG"
                strength = min(long_score, 95)
            elif short_score >= 50:
                signal_type = "SHORT"
                strength = min(short_score, 95)
            else:
                return None
            
            # Calculate targets
            if signal_type == "LONG":
                entry1 = price
                entry2 = price * 0.99
                target1 = price * 1.03
                target2 = price * 1.06
                target3 = price * 1.12
                stop_loss = price * 0.98
            else:  # SHORT
                entry1 = price
                entry2 = price * 1.01
                target1 = price * 0.97
                target2 = price * 0.94
                target3 = price * 0.88
                stop_loss = price * 1.02
            
            return {
                'symbol': symbol_data['symbol'].replace('/USDT', ''),
                'type': signal_type,
                'strength': strength,
                'price': price,
                'entry1': entry1,
                'entry2': entry2,
                'target1': target1,
                'target2': target2,
                'target3': target3,
                'stop_loss': stop_loss,
                'rsi': rsi,
                'volume_24h': symbol_data['volume'],
                'price_change_24h': symbol_data['change'],
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating signal: {str(e)}")
            return None
    
    def run_analysis(self):
        """Run complete real analysis"""
        try:
            logger.info("🚀 Starting REAL crypto analysis of top 200...")
            
            # Get top symbols
            top_symbols = self.get_top_symbols(config.CRYPTO_COUNT)
            if not top_symbols:
                return []
            
            signals = []
            
            for i, symbol_data in enumerate(top_symbols):  # Analyze all top symbols (up to config.CRYPTO_COUNT)
                try:
                    symbol = symbol_data['symbol']
                    logger.info(f"📊 Analyzing {symbol} ({i+1}/{len(top_symbols)})")
                    
                    # Get OHLCV data
                    df = self.get_ohlcv_data(symbol)
                    if df is None or len(df) < 30:
                        continue
                    
                    # Calculate indicators
                    df = self.calculate_indicators(df)
                    
                    # Generate signal
                    signal = self.generate_signal(symbol_data, df)
                    if signal and signal['strength'] >= 60:
                        signals.append(signal)
                    
                    # Rate limiting - optimized for 200 cryptos  
                    time.sleep(0.12)  # Safe rate for 200 symbols
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Sort by strength
            signals.sort(key=lambda x: x['strength'], reverse=True)
            top_signals = signals[:config.SIGNAL_COUNT]
            
            logger.info(f"✅ Generated {len(top_signals)} high-quality signals")
            return top_signals
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return []
    
    def send_telegram_message(self, signals):
        """Send analysis results to Telegram"""
        try:
            if not signals:
                message = """🤖 <b>Crypto Futures Analysis</b>
📅 {}

👤 <b>{} {}</b>
📱 {}

📊 <b>Market Status:</b>
- Analyzed: {} top cryptocurrencies
❌ No strong signals found in current market conditions
⏰ Next analysis in 2 hours

🔍 <i>Real-time analysis of top cryptocurrencies</i>""".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    config.USER_INFO['first_name'],
                    config.USER_INFO['last_name'],
                    config.USER_INFO['username'],
                    config.CRYPTO_COUNT
                )
            else:
                message = f"""🚀 <b>Crypto Futures Analysis</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

👤 <b>{config.USER_INFO['first_name']} {config.USER_INFO['last_name']}</b>
📱 {config.USER_INFO['username']}

📊 <b>Market Summary:</b>
- Analyzed: {config.CRYPTO_COUNT} top cryptocurrencies
- Strong Signals: {len(signals)}
- Average Strength: {sum(s['strength'] for s in signals) / len(signals):.0f}%

🏆 <b>Top Trading Signals:</b>

"""
                
                for i, signal in enumerate(signals, 1):
                    direction = "🟢" if signal['type'] == "LONG" else "🔴"
                    message += f"""{i}. {direction} <b>{signal['symbol']}/USDT {signal['type']}</b>
💎 Strength: {signal['strength']:.0f}% | RSI: {signal['rsi']:.1f}
💰 Entry: ${signal['entry1']:.4f} - ${signal['entry2']:.4f}
🎯 Targets: ${signal['target1']:.4f} | ${signal['target2']:.4f} | ${signal['target3']:.4f}
🛡️ Stop Loss: ${signal['stop_loss']:.4f}

"""
                
                message += f"⚡ <i>Real-time analysis of top {config.CRYPTO_COUNT} cryptos | Next update in 2 hours</i>"
            
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=15)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"❌ Telegram failed: {str(e)}")
            return False

# Global analyzer instance
analyzer = RealCryptoAnalyzer()

@app.route('/')
def index():
    """Home page"""
    return jsonify({
        "status": "✅ Real Crypto Futures Analysis Bot is running!",
        "service": "crypto-futures-bot",
        "version": "real-2.0",
        "features": [
            "Real-time data from top 200 cryptocurrencies",
            "Technical analysis (RSI, MACD, Bollinger Bands)",
            "Automated Telegram reports",
            "Smart signal generation"
        ],
        "endpoints": {
            "/analyze": "Trigger real analysis",
            "/status": "Check bot status",
            "/test": "Simple test"
        }
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    try:
        logger.info("🧪 Test endpoint called")
        return jsonify({
            "status": "success",
            "message": "✅ Real analysis bot working!",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "config": {
                "user": config.USER_INFO['username'],
                "chat_id": config.USER_INFO['chat_id'],
                "crypto_count": config.CRYPTO_COUNT,
                "analysis_ready": True
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/analyze')
def analyze():
    """Real crypto analysis endpoint"""
    try:
        logger.info("🚀 Real analysis triggered via HTTP")
        
        # Run real analysis
        signals = analyzer.run_analysis()
        
        # Send to Telegram
        telegram_sent = analyzer.send_telegram_message(signals)
        
        return jsonify({
            "status": "success",
            "message": "✅ Real analysis completed!",
            "signals_found": len(signals),
            "signals": signals,
            "telegram_sent": telegram_sent,
            "analysis_type": "real",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"❌ Real analysis failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"❌ Real analysis failed: {str(e)}"
        }), 500

@app.route('/status')
def status():
    """Check bot status"""
    return jsonify({
        "status": "running",
        "service": "crypto-futures-bot",
        "version": "real-2.0",
        "analysis_type": "real-time",
        "features": {
            "real_data": True,
            "technical_analysis": True,
            "telegram_integration": True,
            "automated_signals": True
        },
        "config": {
            "crypto_count": config.CRYPTO_COUNT,
            "signal_count": config.SIGNAL_COUNT,
            "telegram_user": config.USER_INFO['username']
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def main():
    """Main function"""
    logger.info("🤖 Real Crypto Analysis Bot Starting...")
    logger.info(f"📱 Telegram: {config.USER_INFO['username']}")
    logger.info("🌐 HTTP service ready for real analysis...")
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main()
