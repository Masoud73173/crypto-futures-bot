#!/usr/bin/env python3
"""
Crypto Futures Analysis Bot - Simple HTTP Version
"""

from flask import Flask, jsonify
import requests
import json
import logging
from datetime import datetime
import os
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    """Home page"""
    return jsonify({
        "status": "✅ Crypto Futures Analysis Bot is running!",
        "service": "crypto-futures-bot",
        "version": "1.0",
        "endpoints": {
            "/analyze": "Trigger manual analysis",
            "/status": "Check bot status",
            "/test": "Simple test endpoint"
        }
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    try:
        logger.info("🧪 Test endpoint called")
        
        # Simple test without external APIs
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return jsonify({
            "status": "success",
            "message": "✅ Test completed successfully!",
            "timestamp": current_time,
            "config": {
                "user": config.USER_INFO['username'],
                "chat_id": config.USER_INFO['chat_id'],
                "crypto_count": config.CRYPTO_COUNT
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"❌ Test failed: {str(e)}"
        }), 500

@app.route('/simple-telegram')
def simple_telegram():
    """Send simple telegram message"""
    try:
        logger.info("📱 Simple telegram test")
        
        # Simple telegram message
        message = f"""🤖 <b>Crypto Bot Test</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
👤 <b>{config.USER_INFO['first_name']} {config.USER_INFO['last_name']}</b>
✅ Bot is working properly!"""
        
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': config.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return jsonify({
                "status": "success",
                "message": "✅ Telegram message sent successfully!",
                "telegram_response": response.json()
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"❌ Telegram error: {response.status_code}",
                "response": response.text
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Telegram test failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"❌ Telegram test failed: {str(e)}"
        }), 500

@app.route('/analyze')
def analyze():
    """Simple analysis without heavy processing"""
    try:
        logger.info("🚀 Simple analysis triggered")
        
        # Mock analysis data instead of real API calls
        mock_signals = [
            {
                "symbol": "BTC",
                "type": "LONG",
                "strength": 85,
                "price": 67500,
                "target1": 69500,
                "target2": 72000,
                "stop_loss": 65000
            },
            {
                "symbol": "ETH", 
                "type": "SHORT",
                "strength": 78,
                "price": 3450,
                "target1": 3300,
                "target2": 3150,
                "stop_loss": 3600
            }
        ]
        
        # Create simple telegram message
        message = f"""🚀 <b>Crypto Analysis Test</b>
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

👤 <b>{config.USER_INFO['first_name']} {config.USER_INFO['last_name']}</b>
📱 {config.USER_INFO['username']}

📊 <b>Sample Signals:</b>
1. 🟢 <b>BTC/USDT LONG</b> - Strength: 85%
2. 🔴 <b>ETH/USDT SHORT</b> - Strength: 78%

🤖 <i>Simple Analysis Bot Test</i>"""

        # Send telegram message
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': config.TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        telegram_response = requests.post(url, json=payload, timeout=10)
        
        return jsonify({
            "status": "success",
            "message": "✅ Simple analysis completed!",
            "signals_count": len(mock_signals),
            "signals": mock_signals,
            "telegram_sent": telegram_response.status_code == 200,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"❌ Analysis failed: {str(e)}"
        }), 500

@app.route('/status')
def status():
    """Check bot status"""
    return jsonify({
        "status": "running",
        "service": "crypto-futures-bot",
        "version": "simple-1.0",
        "config": {
            "crypto_count": config.CRYPTO_COUNT,
            "report_interval": f"{config.REPORT_INTERVAL_HOURS} hours",
            "telegram_user": config.USER_INFO['username'],
            "outputs": config.OUTPUT_SETTINGS
        },
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def main():
    """Main function to run the Flask app"""
    logger.info("🤖 Simple Crypto Bot Starting...")
    logger.info(f"📱 Telegram: {config.USER_INFO['username']} ({config.USER_INFO['chat_id']})")
    logger.info("🌐 Starting HTTP service on port 8080...")
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    main()
