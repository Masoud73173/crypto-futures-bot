#!/usr/bin/env python3
"""
Crypto Futures Analysis Bot
Advanced Trading Signal Generator with HTML Reports and Telegram Integration
Created for automated cryptocurrency futures analysis
"""

import ccxt
import pandas as pd
import numpy as np
import ta
import requests
import schedule
import time
import logging
import json
from datetime import datetime, timedelta
import pytz
import asyncio
import aiohttp
from typing import List, Dict, Tuple, Optional
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CryptoAnalyzer:
    """Main class for cryptocurrency futures analysis"""
    
    def __init__(self):
        """Initialize the analyzer with exchange and settings"""
        self.exchange = getattr(ccxt, config.EXCHANGE)({
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        self.signals = []
        self.market_data = {}
        self.analysis_time = None
        
    async def fetch_top_cryptocurrencies(self) -> List[str]:
        """Fetch top cryptocurrencies by market cap"""
        try:
            markets = self.exchange.load_markets()
            futures_symbols = [symbol for symbol in markets.keys() 
                             if '/USDT' in symbol and markets[symbol]['type'] == 'future']
            
            # Get 24h ticker data for volume filtering
            tickers = self.exchange.fetch_tickers()
            
            # Filter by volume and sort by volume
            valid_symbols = []
            for symbol in futures_symbols[:150]:  # Check more symbols to get best ones
                if symbol in tickers:
                    ticker = tickers[symbol]
                    if ticker['quoteVolume'] and ticker['quoteVolume'] > config.MIN_VOLUME_24H:
                        valid_symbols.append((symbol, ticker['quoteVolume']))
            
            # Sort by volume and take top 100
            valid_symbols.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [symbol[0] for symbol in valid_symbols[:config.CRYPTO_COUNT]]
            
            logger.info(f"✅ Fetched {len(top_symbols)} top cryptocurrencies")
            return top_symbols
            
        except Exception as e:
            logger.error(f"❌ Error fetching cryptocurrencies: {str(e)}")
            # Fallback to manual list
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
            ]
    
    async def fetch_ohlcv_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the dataframe"""
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=config.RSI_PERIOD).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'], 
                                window_fast=config.MACD_FAST,
                                window_slow=config.MACD_SLOW, 
                                window_sign=config.MACD_SIGNAL)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], 
                                            window=config.BB_PERIOD, 
                                            window_dev=config.BB_STD)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # Moving Averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
            
            # Volume indicators
            df['volume_sma'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume'], window=20).volume_sma()
            
            # Price change
            df['price_change_24h'] = df['close'].pct_change(24) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {str(e)}")
            return df
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal based on technical analysis"""
        try:
            if len(df) < 30:  # Need enough data
                return None
                
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Get current values
            price = latest['close']
            rsi = latest['rsi']
            macd = latest['macd']
            macd_signal = latest['macd_signal']
            bb_upper = latest['bb_upper']
            bb_lower = latest['bb_lower']
            volume = latest['volume']
            avg_volume = latest['volume_sma']
            price_change = latest['price_change_24h']
            
            # Skip if insufficient price change
            if abs(price_change) < config.MIN_PRICE_CHANGE:
                return None
            
            signal_type = None
            strength = 0
            reasons = []
            
            # LONG Signal Conditions
            long_conditions = 0
            if rsi < 40 and rsi > 25:  # Oversold but not extreme
                long_conditions += 2
                reasons.append("RSI oversold")
            
            if macd > macd_signal and prev['macd'] <= prev['macd_signal']:  # MACD crossover
                long_conditions += 3
                reasons.append("MACD bullish crossover")
            
            if price <= bb_lower * 1.02:  # Near lower Bollinger Band
                long_conditions += 2
                reasons.append("Price near BB lower")
            
            if latest['ema_12'] > latest['ema_26']:  # EMA alignment
                long_conditions += 1
                reasons.append("EMA bullish")
            
            if volume > avg_volume * 1.2:  # Volume confirmation
                long_conditions += 1
                reasons.append("High volume")
            
            # SHORT Signal Conditions
            short_conditions = 0
            if rsi > 60 and rsi < 75:  # Overbought but not extreme
                short_conditions += 2
                reasons.append("RSI overbought")
            
            if macd < macd_signal and prev['macd'] >= prev['macd_signal']:  # MACD crossover
                short_conditions += 3
                reasons.append("MACD bearish crossover")
            
            if price >= bb_upper * 0.98:  # Near upper Bollinger Band
                short_conditions += 2
                reasons.append("Price near BB upper")
            
            if latest['ema_12'] < latest['ema_26']:  # EMA alignment
                short_conditions += 1
                reasons.append("EMA bearish")
            
            if volume > avg_volume * 1.2:  # Volume confirmation
                short_conditions += 1
                reasons.append("High volume")
            
            # Determine signal
            if long_conditions >= 4:
                signal_type = "LONG"
                strength = min(long_conditions * 10, 95)
            elif short_conditions >= 4:
                signal_type = "SHORT"
                strength = min(short_conditions * 10, 95)
            else:
                return None
            
            # Calculate entry and targets
            if signal_type == "LONG":
                entry1 = price
                entry2 = price * 0.99
                target1 = price * (1 + config.TAKE_PROFIT_LEVELS[0] / 100)
                target2 = price * (1 + config.TAKE_PROFIT_LEVELS[1] / 100)
                target3 = price * (1 + config.TAKE_PROFIT_LEVELS[2] / 100)
                stop_loss = price * (1 - config.STOP_LOSS_PERCENTAGE / 100)
            else:  # SHORT
                entry1 = price
                entry2 = price * 1.01
                target1 = price * (1 - config.TAKE_PROFIT_LEVELS[0] / 100)
                target2 = price * (1 - config.TAKE_PROFIT_LEVELS[1] / 100)
                target3 = price * (1 - config.TAKE_PROFIT_LEVELS[2] / 100)
                stop_loss = price * (1 + config.STOP_LOSS_PERCENTAGE / 100)
            
            signal = {
                'symbol': symbol.replace('/USDT', ''),
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
                'volume_24h': volume * price,
                'price_change_24h': price_change,
                'reasons': reasons,
                'timestamp': datetime.now(pytz.UTC).strftime(config.DATE_FORMAT)
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {symbol}: {str(e)}")
            return None
    
    async def analyze_market(self) -> List[Dict]:
        """Analyze the entire market and generate signals"""
        logger.info("🚀 Starting market analysis...")
        self.analysis_time = datetime.now(pytz.UTC)
        
        # Fetch top cryptocurrencies
        symbols = await self.fetch_top_cryptocurrencies()
        signals = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"📊 Analyzing {symbol} ({i+1}/{len(symbols)})")
                
                # Fetch OHLCV data
                df = await self.fetch_ohlcv_data(symbol, config.TIMEFRAME)
                if df is None or len(df) < 30:
                    continue
                
                # Calculate technical indicators
                df = self.calculate_technical_indicators(df)
                
                # Generate signal
                signal = self.generate_signal(symbol, df)
                if signal and signal['strength'] >= 50:  # Only strong signals
                    signals.append(signal)
                
                # Add small delay to respect rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"⚠️ Error analyzing {symbol}: {str(e)}")
                continue
        
        # Sort by strength and take top signals
        signals.sort(key=lambda x: x['strength'], reverse=True)
        top_signals = signals[:config.SIGNAL_COUNT]
        
        logger.info(f"✅ Generated {len(top_signals)} high-quality signals")
        return top_signals
    
    def generate_html_report(self, signals: List[Dict]) -> str:
        """Generate beautiful HTML report"""
        try:
            total_analyzed = config.CRYPTO_COUNT
            timestamp = self.analysis_time.strftime(config.DATE_FORMAT)
            
            # Calculate success rate (simulated based on signal strength)
            avg_strength = sum(s['strength'] for s in signals) / len(signals) if signals else 0
            
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.HTML_TITLE}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2C3E50 0%, #3498DB 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 20px;
        }}
        
        .user-info {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            display: inline-block;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #2C3E50;
            margin-bottom: 5px;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9rem;
        }}
        
        .signals-section {{
            padding: 30px;
        }}
        
        .section-title {{
            font-size: 1.8rem;
            margin-bottom: 20px;
            color: #2C3E50;
            text-align: center;
            position: relative;
        }}
        
        .section-title:after {{
            content: '';
            position: absolute;
            bottom: -5px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 3px;
            background: #3498DB;
            border-radius: 2px;
        }}
        
        .signal-card {{
            background: white;
            margin-bottom: 20px;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border-left: 5px solid;
        }}
        
        .signal-long {{
            border-left-color: #27AE60;
        }}
        
        .signal-short {{
            border-left-color: #E74C3C;
        }}
        
        .signal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .signal-symbol {{
            font-size: 1.4rem;
            font-weight: bold;
            color: #2C3E50;
        }}
        
        .signal-type {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
            font-size: 0.9rem;
        }}
        
        .type-long {{
            background: #27AE60;
        }}
        
        .type-short {{
            background: #E74C3C;
        }}
        
        .signal-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .detail-item {{
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .detail-label {{
            font-size: 0.8rem;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        
        .detail-value {{
            font-weight: bold;
            color: #2C3E50;
        }}
        
        .success-rate {{
            text-align: center;
            font-size: 0.9rem;
            color: #27AE60;
            font-weight: bold;
        }}
        
        .footer {{
            background: #2C3E50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 2rem;
            }}
            
            .stats {{
                grid-template-columns: 1fr;
            }}
            
            .signal-header {{
                flex-direction: column;
                gap: 10px;
            }}
            
            .signal-details {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 {config.HTML_TITLE}</h1>
            <div class="subtitle">📅 {timestamp}</div>
            <div class="user-info">
                <div><strong>👤 {config.USER_INFO['first_name']} {config.USER_INFO['last_name']}</strong></div>
                <div>📱 {config.USER_INFO['username']}</div>
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{total_analyzed}</div>
                <div class="stat-label">Cryptos Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(signals)}</div>
                <div class="stat-label">Signals Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{avg_strength:.0f}%</div>
                <div class="stat-label">Average Strength</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{config.REPORT_INTERVAL_HOURS}h</div>
                <div class="stat-label">Update Interval</div>
            </div>
        </div>
        
        <div class="signals-section">
            <h2 class="section-title">🏆 Top Trading Signals</h2>
"""
            
            # Add signals
            for i, signal in enumerate(signals, 1):
                signal_class = "signal-long" if signal['type'] == "LONG" else "signal-short"
                type_class = "type-long" if signal['type'] == "LONG" else "type-short"
                
                html_content += f"""
            <div class="signal-card {signal_class}">
                <div class="signal-header">
                    <div class="signal-symbol">{i}. 🟢 {signal['symbol']}/USDT {signal['type']}</div>
                    <div class="signal-type {type_class}">{signal['type']}</div>
                </div>
                
                <div class="signal-details">
                    <div class="detail-item">
                        <div class="detail-label">Entry 1</div>
                        <div class="detail-value">${signal['entry1']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Entry 2</div>
                        <div class="detail-value">${signal['entry2']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Target 1 (+{config.TAKE_PROFIT_LEVELS[0]}%)</div>
                        <div class="detail-value">${signal['target1']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Target 2 (+{config.TAKE_PROFIT_LEVELS[1]}%)</div>
                        <div class="detail-value">${signal['target2']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Target 3 (+{config.TAKE_PROFIT_LEVELS[2]}%)</div>
                        <div class="detail-value">${signal['target3']:.4f}</div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">Stop Loss</div>
                        <div class="detail-value">${signal['stop_loss']:.4f}</div>
                    </div>
                </div>
                
                <div class="success-rate">💎 Signal Strength: {signal['strength']:.0f}% | RSI: {signal['rsi']:.1f}</div>
            </div>
"""
            
            html_content += """
        </div>
        
        <div class="footer">
            <div>⚡ Powered by Advanced Technical Analysis | 🤖 Automated Crypto Futures Bot</div>
            <div style="margin-top: 10px; font-size: 0.8rem; opacity: 0.8;">
                📊 Analysis includes RSI, MACD, Bollinger Bands, Moving Averages & Volume Analysis
            </div>
        </div>
    </div>
</body>
</html>
"""
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Error generating HTML report: {str(e)}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    async def send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram"""
        try:
            url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                'chat_id': config.TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Telegram message sent successfully")
                        return True
                    else:
                        logger.error(f"❌ Telegram error: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Error sending telegram message: {str(e)}")
            return False
    
    def format_telegram_message(self, signals: List[Dict]) -> str:
        """Format message for Telegram"""
        timestamp = self.analysis_time.strftime(config.DATE_FORMAT)
        
        message = f"""🚀 <b>Crypto Futures Analysis Report</b>
📅 {timestamp}

👤 <b>{config.USER_INFO['first_name']} {config.USER_INFO['last_name']}</b>
📱 {config.USER_INFO['username']}

📊 <b>Market Summary:</b>
- Analyzed: {config.CRYPTO_COUNT} cryptos
- Signals Generated: {len(signals)}
- Update Interval: {config.REPORT_INTERVAL_HOURS} hours

🏆 <b>Top Trading Signals:</b>

"""
        
        for i, signal in enumerate(signals[:10], 1):  # Top 10 for Telegram
            direction_emoji = "🟢" if signal['type'] == "LONG" else "🔴"
            
            message += f"""{i}. {direction_emoji} <b>{signal['symbol']}/USDT {signal['type']}</b>
- Signal Strength: {signal['strength']:.0f}%
- Entry 1: ${signal['entry1']:.4f}
- Entry 2: ${signal['entry2']:.4f}
- Target 1 (+{config.TAKE_PROFIT_LEVELS[0]}%): ${signal['target1']:.4f}
- Target 2 (+{config.TAKE_PROFIT_LEVELS[1]}%): ${signal['target2']:.4f}
- Target 3 (+{config.TAKE_PROFIT_LEVELS[2]}%): ${signal['target3']:.4f}
- Stop Loss: ${signal['stop_loss']:.4f}

"""
        
        message += f"""⚡ <b>Technical Analysis:</b>
- RSI, MACD, Bollinger Bands
- Volume & Moving Averages
- Risk Management: {config.STOP_LOSS_PERCENTAGE}% Stop Loss

🤖 <i>Automated Crypto Futures Analysis Bot</i>"""
        
        return message
    
    async def run_analysis(self):
        """Run complete analysis cycle"""
        try:
            logger.info("🎯 Starting analysis cycle...")
            
            # Analyze market and generate signals
            signals = await self.analyze_market()
            
            if not signals:
                logger.warning("⚠️ No signals generated")
                return
            
            # Generate HTML report
            if config.OUTPUT_SETTINGS['html']:
                logger.info("📄 Generating HTML report...")
                html_report = self.generate_html_report(signals)
                with open(config.HTML_OUTPUT_PATH, 'w', encoding='utf-8') as f:
                    f.write(html_report)
                logger.info(f"✅ HTML report saved: {config.HTML_OUTPUT_PATH}")
            
            # Send Telegram message
            if config.OUTPUT_SETTINGS['telegram']:
                logger.info("📱 Sending Telegram report...")
                telegram_message = self.format_telegram_message(signals)
                await self.send_telegram_message(telegram_message)
            
            logger.info(f"🎉 Analysis complete! Generated {len(signals)} signals")
            
        except Exception as e:
            logger.error(f"❌ Error in analysis cycle: {str(e)}")

def main():
    """Main function to run the bot"""
    logger.info("🤖 Crypto Futures Analysis Bot Starting...")
    logger.info(f"📊 Configuration: {config.CRYPTO_COUNT} cryptos every {config.REPORT_INTERVAL_HOURS} hours")
    logger.info(f"📱 Telegram: {config.USER_INFO['username']} ({config.USER_INFO['chat_id']})")
    
    analyzer = CryptoAnalyzer()
    
    # Schedule the analysis
    schedule.every(config.REPORT_INTERVAL_HOURS).hours.do(
        lambda: asyncio.run(analyzer.run_analysis())
    )
    
    # Run initial analysis
    logger.info("🚀 Running initial analysis...")
    asyncio.run(analyzer.run_analysis())
    
    # Keep the bot running
    logger.info(f"⏰ Bot scheduled to run every {config.REPORT_INTERVAL_HOURS} hours")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()
