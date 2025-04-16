import sys
import os
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from engine import SignalEngine
from config import TELEGRAM_BOT_TOKEN
import ccxt.async_support as ccxt
import pandas as pd
import logging
from datetime import datetime, timedelta
import market_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_data(exchange, asset: str):
    try:
        ohlcv = await exchange.fetch_ohlcv(asset, timeframe='5m', limit=100)
        return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        logger.error(f"Failed to fetch data for {asset} from {exchange.id}: {str(e)}")
        return None
    finally:
        await exchange.close()  # Ensure session closes

engine = SignalEngine()

async def debug_update(update, context):
    logger.info(f"Received update: {update}")

async def start_command(update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! I'm your trading bot. Use /signal to get a trading signal.")

async def signal_command(update, context):
    logger.info("Received /signal command")
    
    # Full Pocket Options OTC token list as USDT pairs
    assets = ["AVAX/USDT", "BTC/USDT", "BNB/USDT", "ETH/USDT", "MATIC/USDT", 
              "LINK/USDT", "LTC/USDT", "TRX/USDT", "DOT/USDT", "DOGE/USDT", 
              "SOL/USDT", "TON/USDT", "DASH/USDT"]
    exchange = ccxt.kucoin()  # Using KuCoin for max flexibility
    
    # Fetch data and generate signals
    best_signal = None
    best_confidence = 0
    best_asset = None
    best_data = None

    for asset in assets:
        data = await fetch_data(exchange, asset)
        if data is None or len(data) < 50:
            continue
        signal = await engine.process_market_data(asset, data)
        if signal and signal['confidence'] > best_confidence:
            best_signal = signal
            best_confidence = signal['confidence']
            best_asset = asset
            best_data = data

    if best_signal is None:
        logger.info("No signal generated for any asset")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No signal generated for any asset.")
        return

    # Calculate ATR and trade duration
    atr = market_analysis.calculate_atr(best_data)
    trade_duration = market_analysis.determine_trade_duration(atr)
    best_signal['trade_duration'] = trade_duration

    # Send warning and wait
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Preparing signal for {best_asset}... Please select the pair on Pocket Options.")
    await asyncio.sleep(5)

    # Set start time
    start_time = datetime.now() + timedelta(seconds=5)
    best_signal['start_time'] = start_time.strftime("%H:%M:%S")
    
    # Send signal
    msg = engine.format_signal_message(best_signal)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)

async def feedback_command(update, context, outcome: bool):
    if not context.args:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please provide a signal ID (e.g., /won 12345)")
        return
    signal_id = context.args[0]
    await engine.process_feedback(signal_id, outcome)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Feedback recorded: {'win' if outcome else 'loss'} for {signal_id}")

print(f"TELEGRAM_BOT_TOKEN in main.py: {TELEGRAM_BOT_TOKEN}")
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start_command))
app.add_handler(CommandHandler("signal", signal_command))
app.add_handler(CommandHandler("won", lambda u, c: feedback_command(u, c, True)))
app.add_handler(CommandHandler("lost", lambda u, c: feedback_command(u, c, False)))
app.add_handler(MessageHandler(filters.ALL, debug_update))

while True:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app.run_polling(timeout=10)
    except Exception as e:
        logger.error(f"Polling failed, restarting: {str(e)}")
        loop.close()
        asyncio.sleep(5)