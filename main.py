# main.py - PREMIER FOREX AI QUANT V2.5 (Render Optimized with Performance Tracking)

import os
import ccxt
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from telegram import Bot
from flask import Flask, jsonify, render_template_string
import threading
import time
import traceback 
import json

# --- ML Imports (Optional - commented out to avoid dependency issues) ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler 

# --- CONFIGURATION ---
from dotenv import load_dotenv 
load_dotenv() 

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FOREX_PAIRS = [p.strip() for p in os.getenv("FOREX_PAIRS", "EUR/USD,GBP/USD,USD/JPY,AUD/USD,USD/CHF").split(',')]
TIMEFRAME_MAIN = "4h"  # Major Trend
TIMEFRAME_ENTRY = "1h" # Entry Precision

# Initialize Bot and Exchange (Defaulting to Kraken for India/Forex Stability)
bot = Bot(token=TELEGRAM_BOT_TOKEN)
exchange = ccxt.kraken({
    'enableRateLimit': True, 
    'rateLimit': 2000,
    'params': {'timeout': 20000} # Explicit 20s timeout
})

bot_stats = {
    "status": "initializing",
    "total_analyses": 0,
    "last_analysis": None,
    "monitored_assets": FOREX_PAIRS,
    "uptime_start": datetime.now().isoformat(),
    "version": "V2.5 Forex Elite Quant + Performance Tracker"
}

# === TRADE TRACKING SYSTEM ===
trade_history = []  # Stores all trades with their outcomes
TRADE_HISTORY_FILE = "forex_trade_history.json"

def load_trade_history():
    """Load trade history from file on startup."""
    global trade_history
    try:
        if os.path.exists(TRADE_HISTORY_FILE):
            with open(TRADE_HISTORY_FILE, 'r') as f:
                trade_history = json.load(f)
                print(f"📊 Loaded {len(trade_history)} historical forex trades")
    except Exception as e:
        print(f"⚠️ Could not load trade history: {e}")
        trade_history = []

def save_trade_history():
    """Save trade history to file."""
    try:
        with open(TRADE_HISTORY_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save trade history: {e}")

def add_trade(symbol, signal, entry_price, tp1, tp2, sl, trend_4h, trend_1h):
    """Add a new trade to tracking system."""
    trade = {
        "id": len(trade_history) + 1,
        "symbol": symbol,
        "signal": signal,
        "entry_price": entry_price,
        "tp1": tp1,
        "tp2": tp2,
        "sl": sl,
        "trend_4h": trend_4h,
        "trend_1h": trend_1h,
        "timestamp": datetime.now().isoformat(),
        "status": "ACTIVE",  # ACTIVE, TP1_HIT, TP2_HIT, SL_HIT
        "outcome": None,  # Will be: "WIN", "LOSS", "PARTIAL_WIN"
        "profit_loss_pips": 0.0,
        "profit_loss_pct": 0.0
    }
    trade_history.append(trade)
    save_trade_history()
    return trade

def calculate_pips(symbol, price1, price2):
    """Calculate pip difference for forex pairs."""
    pip_value = 0.01 if 'JPY' in symbol else 0.0001
    return abs(price2 - price1) / pip_value

def check_trade_outcomes():
    """Check all active trades and update their status."""
    global trade_history
    updated = False
    
    for trade in trade_history:
        if trade['status'] == 'ACTIVE':
            try:
                # Fetch current price
                df = fetch_data_safe(trade['symbol'], '1h')
                if df.empty:
                    continue
                
                current_price = df.iloc[-1]['close']
                entry = trade['entry_price']
                is_buy = "BUY" in trade['signal']
                
                # Check if targets or stop loss hit
                if is_buy:
                    if current_price >= trade['tp2']:
                        trade['status'] = 'TP2_HIT'
                        trade['outcome'] = 'WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], entry, trade['tp2'])
                        trade['profit_loss_pct'] = ((trade['tp2'] - entry) / entry) * 100
                        updated = True
                    elif current_price >= trade['tp1']:
                        trade['status'] = 'TP1_HIT'
                        trade['outcome'] = 'PARTIAL_WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], entry, trade['tp1'])
                        trade['profit_loss_pct'] = ((trade['tp1'] - entry) / entry) * 100
                        updated = True
                    elif current_price <= trade['sl']:
                        trade['status'] = 'SL_HIT'
                        trade['outcome'] = 'LOSS'
                        trade['profit_loss_pips'] = -calculate_pips(trade['symbol'], entry, trade['sl'])
                        trade['profit_loss_pct'] = ((trade['sl'] - entry) / entry) * 100
                        updated = True
                else:  # SELL
                    if current_price <= trade['tp2']:
                        trade['status'] = 'TP2_HIT'
                        trade['outcome'] = 'WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], trade['tp2'], entry)
                        trade['profit_loss_pct'] = ((entry - trade['tp2']) / entry) * 100
                        updated = True
                    elif current_price <= trade['tp1']:
                        trade['status'] = 'TP1_HIT'
                        trade['outcome'] = 'PARTIAL_WIN'
                        trade['profit_loss_pips'] = calculate_pips(trade['symbol'], trade['tp1'], entry)
                        trade['profit_loss_pct'] = ((entry - trade['tp1']) / entry) * 100
                        updated = True
                    elif current_price >= trade['sl']:
                        trade['status'] = 'SL_HIT'
                        trade['outcome'] = 'LOSS'
                        trade['profit_loss_pips'] = -calculate_pips(trade['symbol'], trade['sl'], entry)
                        trade['profit_loss_pct'] = ((entry - trade['sl']) / entry) * 100
                        updated = True
                        
            except Exception as e:
                print(f"Error checking forex trade {trade['id']}: {e}")
    
    if updated:
        save_trade_history()

def generate_daily_report():
    """Generate and send 24-hour forex performance report."""
    try:
        # First, update all trade outcomes
        check_trade_outcomes()
        
        # Get trades from last 24 hours
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_trades = [
            t for t in trade_history 
            if datetime.fromisoformat(t['timestamp']) >= last_24h
        ]
        
        if not recent_trades:
            message = (
                f"╔════════════════════════════════╗\n"
                f"  📊 <b>24-HOUR FOREX REPORT</b>\n"
                f"╚════════════════════════════════╝\n\n"
                f"<b>Period:</b> {last_24h.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}\n\n"
                f"⚠️ No forex signals executed in the last 24 hours.\n\n"
                f"<i>Market monitoring continues...</i>\n"
                f"<i>Next report: {(now + timedelta(hours=24)).strftime('%d %b %H:%M')}</i>"
            )
        else:
            # Calculate statistics
            total_trades = len(recent_trades)
            wins = len([t for t in recent_trades if t['outcome'] == 'WIN'])
            partial_wins = len([t for t in recent_trades if t['outcome'] == 'PARTIAL_WIN'])
            losses = len([t for t in recent_trades if t['outcome'] == 'LOSS'])
            active = len([t for t in recent_trades if t['status'] == 'ACTIVE'])
            
            total_profit_pips = sum([t['profit_loss_pips'] for t in recent_trades if t['outcome'] in ['WIN', 'PARTIAL_WIN']])
            total_loss_pips = sum([abs(t['profit_loss_pips']) for t in recent_trades if t['outcome'] == 'LOSS'])
            net_pips = total_profit_pips - total_loss_pips
            
            total_profit_pct = sum([t['profit_loss_pct'] for t in recent_trades if t['outcome'] in ['WIN', 'PARTIAL_WIN']])
            total_loss_pct = sum([abs(t['profit_loss_pct']) for t in recent_trades if t['outcome'] == 'LOSS'])
            net_pct = total_profit_pct - total_loss_pct
            
            win_rate = ((wins + partial_wins) / (wins + partial_wins + losses) * 100) if (wins + partial_wins + losses) > 0 else 0
            
            # Build detailed message
            message = (
                f"╔════════════════════════════════╗\n"
                f"  📊 <b>24-HOUR FOREX REPORT</b>\n"
                f"╚════════════════════════════════╝\n\n"
                f"<b>📅 Period:</b> {last_24h.strftime('%d %b %H:%M')} - {now.strftime('%d %b %H:%M')}\n\n"
                f"<b>📈 TRADING STATISTICS:</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• Total Signals: <b>{total_trades}</b>\n"
                f"• ✅ Full Wins (TP2): <b>{wins}</b>\n"
                f"• ⚡ Partial Wins (TP1): <b>{partial_wins}</b>\n"
                f"• ❌ Losses (SL): <b>{losses}</b>\n"
                f"• ⏳ Active Trades: <b>{active}</b>\n\n"
                f"<b>💰 PROFIT/LOSS ANALYSIS:</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• Win Rate: <code>{win_rate:.1f}%</code>\n"
                f"• Total Profit: <code>+{total_profit_pips:.1f} pips</code>\n"
                f"• Total Loss: <code>-{total_loss_pips:.1f} pips</code>\n"
                f"• <b>Net P/L: <code>{'🟢 +' if net_pips >= 0 else '🔴 '}{net_pips:.1f} pips</code></b>\n"
                f"• Percentage: <code>{'🟢 +' if net_pct >= 0 else '🔴 '}{net_pct:.2f}%</code>\n\n"
            )
            
            # Add top trades
            closed_trades = [t for t in recent_trades if t['outcome'] in ['WIN', 'PARTIAL_WIN', 'LOSS']]
            if closed_trades:
                message += f"<b>🏆 TOP PERFORMING TRADES:</b>\n"
                message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                sorted_trades = sorted(closed_trades, key=lambda x: x['profit_loss_pips'], reverse=True)[:3]
                for i, t in enumerate(sorted_trades, 1):
                    emoji = "🟢" if t['profit_loss_pips'] > 0 else "🔴"
                    message += f"{i}. {t['symbol']} - {emoji} {t['profit_loss_pips']:+.1f} pips ({t['outcome']})\n"
                message += "\n"
            
            # Add active trades info
            if active > 0:
                message += f"<b>⏳ ACTIVE POSITIONS:</b>\n"
                message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                active_trades = [t for t in recent_trades if t['status'] == 'ACTIVE']
                for t in active_trades[:5]:  # Show all active (max 5)
                    decimals = 3 if 'JPY' in t['symbol'] else 5
                    message += f"• {t['symbol']} - {t['signal']} @ {t['entry_price']:.{decimals}f}\n"
                message += "\n"
            
            # Performance by pair
            pair_performance = {}
            for t in closed_trades:
                if t['symbol'] not in pair_performance:
                    pair_performance[t['symbol']] = {'pips': 0, 'trades': 0}
                pair_performance[t['symbol']]['pips'] += t['profit_loss_pips']
                pair_performance[t['symbol']]['trades'] += 1
            
            if pair_performance:
                message += f"<b>📊 PERFORMANCE BY PAIR:</b>\n"
                message += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                for pair, data in sorted(pair_performance.items(), key=lambda x: x[1]['pips'], reverse=True):
                    emoji = "🟢" if data['pips'] > 0 else "🔴"
                    message += f"• {pair}: {emoji} {data['pips']:+.1f} pips ({data['trades']} trades)\n"
                message += "\n"
            
            message += (
                f"----------------------------------------\n"
                f"<i>Next report: {(now + timedelta(hours=24)).strftime('%d %b %H:%M')}</i>\n"
                f"<i>Verified AI Forex Analysis V2.5 Elite</i>"
            )
        
        # Send report
        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML'))
        print(f"✅ Daily forex report sent at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Failed to generate daily forex report: {e}")
        traceback.print_exc()

# =========================================================================
# === ADVANCED QUANT LOGIC ===
# =========================================================================

def get_pip_value(pair):
    return 0.01 if 'JPY' in pair else 0.0001

def calculate_cpr_levels(df_daily):
    """Calculates Pivot Points for Institutional Target Setting."""
    if df_daily.empty or len(df_daily) < 2: return None
    prev_day = df_daily.iloc[-2]
    H, L, C = prev_day['high'], prev_day['low'], prev_day['close']
    PP = (H + L + C) / 3.0
    BC = (H + L) / 2.0
    TC = PP - BC + PP
    return {
        'PP': PP, 'TC': TC, 'BC': BC,
        'R1': 2*PP - L, 'S1': 2*PP - H,
        'R2': PP + (H - L), 'S2': PP - (H - L)
    }

def fetch_data_safe(symbol, timeframe):
    """Robust fetcher with retries and symbol ID normalization."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not exchange.markets: exchange.load_markets()
            market_id = exchange.market(symbol)['id']
            ohlcv = exchange.fetch_ohlcv(market_id, timeframe, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['sma9'] = df['close'].rolling(9).mean()
            df['sma20'] = df['close'].rolling(20).mean()
            return df.dropna()
        except Exception:
            if attempt < max_retries - 1: time.sleep(5)
    return pd.DataFrame()

# =========================================================================
# === MULTI-TIMEFRAME CONFLUENCE ENGINE ===
# =========================================================================

def generate_and_send_signal(symbol):
    global bot_stats
    try:
        # 1. Multi-Timeframe Confluence
        df_4h = fetch_data_safe(symbol, TIMEFRAME_MAIN)
        df_1h = fetch_data_safe(symbol, TIMEFRAME_ENTRY)
        
        # 2. Daily Data for CPR Targets
        if not exchange.markets: exchange.load_markets()
        market_id = exchange.market(symbol)['id']
        ohlcv_d = exchange.fetch_ohlcv(market_id, '1d', limit=5)
        df_d = pd.DataFrame(ohlcv_d, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        cpr = calculate_cpr_levels(df_d)

        if df_4h.empty or df_1h.empty or cpr is None: return

        # 3. Analyze Trend Confluence
        price = df_4h.iloc[-1]['close']
        trend_4h = "BULLISH" if df_4h.iloc[-1]['sma9'] > df_4h.iloc[-1]['sma20'] else "BEARISH"
        trend_1h = "BULLISH" if df_1h.iloc[-1]['sma9'] > df_1h.iloc[-1]['sma20'] else "BEARISH"
        
        # 4. Master Logic
        signal = "HOLD / WAIT"
        emoji = "⏳"
        
        if trend_4h == "BULLISH" and trend_1h == "BULLISH" and price > cpr['PP']:
            signal = "STRONG BUY"
            emoji = "🚀"
        elif trend_4h == "BEARISH" and trend_1h == "BEARISH" and price < cpr['PP']:
            signal = "STRONG SELL"
            emoji = "🔻"

        # 5. Risk Management Targets
        is_buy = "BUY" in signal
        tp1 = cpr['R1'] if is_buy else cpr['S1']
        tp2 = cpr['R2'] if is_buy else cpr['S2']
        sl = min(cpr['BC'], cpr['TC']) if is_buy else max(cpr['BC'], cpr['TC'])
        
        decimals = 3 if 'JPY' in symbol else 5

        # === Track trade if it's a BUY/SELL signal ===
        if "BUY" in signal or "SELL" in signal:
            add_trade(symbol, signal, price, tp1, tp2, sl, trend_4h, trend_1h)
            
            # Calculate pip targets
            tp1_pips = calculate_pips(symbol, price, tp1)
            tp2_pips = calculate_pips(symbol, price, tp2)
            sl_pips = calculate_pips(symbol, price, sl)

        # --- PREMIUM SIGNAL TEMPLATE ---
        message = (
            f"╔════════════════════════════════╗\n"
            f"  🌍 <b>PREMIER FOREX AI BOT</b>\n"
            f"╚════════════════════════════════╝\n\n"
            f"<b>Pair:</b> {symbol}\n"
            f"<b>Rate:</b> <code>{price:.{decimals}f}</code>\n\n"
            f"--- 🚨 {emoji} <b>SIGNAL: {signal}</b> 🚨 ---\n\n"
            f"<b>📈 CONFLUENCE ANALYSIS:</b>\n"
            f"• 4H Trend: <code>{trend_4h}</code>\n"
            f"• 1H Trend: <code>{trend_1h}</code>\n"
            f"• Pivot: {'Above' if price > cpr['PP'] else 'Below'} PP\n\n"
        )
        
        if "BUY" in signal or "SELL" in signal:
            message += (
                f"<b>🎯 TARGET LEVELS:</b>\n"
                f"✅ <b>Take Profit 1:</b> <code>{tp1:.{decimals}f}</code> (+{tp1_pips:.1f} pips)\n"
                f"🔥 <b>Take Profit 2:</b> <code>{tp2:.{decimals}f}</code> (+{tp2_pips:.1f} pips)\n"
                f"🛑 <b>Stop Loss:</b> <code>{sl:.{decimals}f}</code> (-{sl_pips:.1f} pips)\n"
                f"<b>Risk/Reward:</b> <code>1:{(tp2_pips/sl_pips):.2f}</code>\n\n"
            )
        
        message += (
            f"----------------------------------------\n"
            f"<i>Verified AI Forex Analysis</i>"
        )

        asyncio.run(bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML'))
        
        bot_stats['total_analyses'] += 1
        bot_stats['last_analysis'] = datetime.now().isoformat()
        bot_stats['status'] = "operational"

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

# =========================================================================
# === GUNICORN-SAFE INITIALIZATION ===
# =========================================================================

def start_bot():
    print(f"🚀 Initializing {bot_stats['version']}...")
    
    # Load historical trades
    load_trade_history()
    
    scheduler = BackgroundScheduler()
    
    # Schedule signal generation (every hour and half-hour)
    for s in FOREX_PAIRS:
        scheduler.add_job(generate_and_send_signal, 'cron', minute='0,30', args=[s])
    
    # === Schedule daily report at 9:00 AM every day ===
    scheduler.add_job(generate_daily_report, 'cron', hour=9, minute=0)
    
    # === Check trade outcomes every 30 minutes ===
    scheduler.add_job(check_trade_outcomes, 'cron', minute='*/30')
    
    scheduler.start()
    
    # Run immediate baseline check in separate threads
    for s in FOREX_PAIRS:
        threading.Thread(target=generate_and_send_signal, args=(s,)).start()

# Start bot outside main block for Gunicorn support
start_bot()

app = Flask(__name__)

@app.route('/')
def home():
    # Calculate quick stats
    total_trades = len(trade_history)
    wins = len([t for t in trade_history if t['outcome'] == 'WIN'])
    losses = len([t for t in trade_history if t['outcome'] == 'LOSS'])
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    total_pips = sum([t.get('profit_loss_pips', 0) for t in trade_history if t.get('outcome') in ['WIN', 'PARTIAL_WIN', 'LOSS']])
    
    return render_template_string("""
        <body style="font-family:sans-serif; background:#020617; color:#f8fafc; text-align:center; padding-top:50px;">
            <div style="background:#0f172a; display:inline-block; padding:40px; border-radius:12px; border:1px solid #1e293b; max-width:600px;">
                <h1 style="color:#38bdf8;">Forex AI Pro Dashboard</h1>
                <p style="font-size:1.2em;">Status: <span style="color:#4ade80;">Active</span></p>
                <hr style="border-color:#1e293b;">
                <div style="text-align:left; margin-top:20px;">
                    <p><b>Analyses Streamed:</b> {{a}}</p>
                    <p><b>Total Trades:</b> {{tt}}</p>
                    <p><b>Win Rate:</b> {{wr:.1f}}%</p>
                    <p><b>Total Pips:</b> {{tp:+.1f}}</p>
                    <p><b>Version:</b> <i>{{v}}</i></p>
                </div>
                <hr style="border-color:#1e293b;">
                <p style="font-size:0.8em; color:#94a3b8;">Last Analysis: {{t}}</p>
            </div>
        </body>
    """, a=bot_stats['total_analyses'], v=bot_stats['version'], t=bot_stats['last_analysis'],
         tt=total_trades, wr=win_rate, tp=total_pips)

@app.route('/health')
def health(): return jsonify({"status": "healthy"}), 200

@app.route('/stats')
def stats():
    """API endpoint for forex trade statistics."""
    wins = len([t for t in trade_history if t['outcome'] == 'WIN'])
    partial = len([t for t in trade_history if t['outcome'] == 'PARTIAL_WIN'])
    losses = len([t for t in trade_history if t['outcome'] == 'LOSS'])
    active = len([t for t in trade_history if t['status'] == 'ACTIVE'])
    total_pips = sum([t.get('profit_loss_pips', 0) for t in trade_history if t.get('outcome') in ['WIN', 'PARTIAL_WIN', 'LOSS']])
    
    return jsonify({
        "total_trades": len(trade_history),
        "wins": wins,
        "partial_wins": partial,
        "losses": losses,
        "active": active,
        "win_rate": (wins + partial) / (wins + partial + losses) * 100 if (wins + partial + losses) > 0 else 0,
        "total_pips": round(total_pips, 1)
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

