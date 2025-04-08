import sys
import os
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from engine import SignalEngine
from config import TELEGRAM_BOT_TOKEN
import ccxt.async_support as ccxt
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_data(asset="BTC/USD"):
    exchanges = [ccxt.kraken(), ccxt.binance()]
    for exchange in exchanges:
        try:
            ohlcv = await exchange.fetch_ohlcv(asset, timeframe='1m', limit=100)  # Changed from 50 to 100
            await exchange.close()
            return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            logger.error(f"Failed to fetch data from {exchange.id}: {str(e)}")
            await exchange.close()
    logger.error("All exchanges failed to fetch data")
    return None

engine = SignalEngine()

# Debug handler to log all incoming updates
async def debug_update(update, context):
    logger.info(f"Received update: {update}")

# Handler for /start command
async def start_command(update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hello! I'm your trading bot. Use /signal to get a trading signal.")

# Updated /signal command with additional logging
async def signal_command(update, context):
    logger.info("Received /signal command")
    data = await fetch_data()
    if data is None:
        logger.error("Failed to fetch market data")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to fetch market data. Try again later.")
        return
    logger.info("Fetched market data successfully")
    signal = await engine.process_market_data("BTC/USD", data)
    if signal:
        price = data['close'].iloc[-1]
        msg = engine.format_signal_message(signal, price)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode='Markdown')
    else:
        logger.info("No signal generated")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="No signal generated.")

async def feedback_command(update, context, outcome: bool):
    if not context.args:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please provide a signal ID (e.g., /won 12345)")
        return
    signal_id = context.args[0]
    await engine.process_feedback(signal_id, outcome)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Feedback recorded: {'win' if outcome else 'loss'} for {signal_id}")

print(f"TELEGRAM_BOT_TOKEN in main.py: {TELEGRAM_BOT_TOKEN}")
app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# Add handlers
app.add_handler(CommandHandler("start", start_command))
app.add_handler(CommandHandler("signal", signal_command))
app.add_handler(CommandHandler("won", lambda u, c: feedback_command(u, c, True)))
app.add_handler(CommandHandler("lost", lambda u, c: feedback_command(u, c, False)))
app.add_handler(MessageHandler(filters.ALL, debug_update))

# Run polling with a restart loop
while True:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app.run_polling(timeout=10)
    except Exception as e:
        logger.error(f"Polling failed, restarting: {str(e)}")
        loop.close()
        asyncio.sleep(5)  # Wait 5 seconds before restarting