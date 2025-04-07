import sys
import os
import asyncio
from telegram.ext import Application, CommandHandler
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
            ohlcv = await exchange.fetch_ohlcv(asset, timeframe='1m', limit=50)
            await exchange.close()
            return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except Exception as e:
            logger.error(f"Failed to fetch data from {exchange.id}: {str(e)}")
            await exchange.close()
    logger.error("All exchanges failed to fetch data")
    return None

engine = SignalEngine()

async def signal_command(update, context):
    data = await fetch_data()
    if data is None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to fetch market data. Try again later.")
        return
    signal = await engine.process_market_data("BTC/USD", data)
    if signal:
        price = data['close'].iloc[-1]
        msg = engine.format_signal_message(signal, price)
        await context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode='Markdown')

async def feedback_command(update, context, outcome: bool):
    if not context.args:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please provide a signal ID (e.g., /won 12345)")
        return
    signal_id = context.args[0]
    await engine.process_feedback(signal_id, outcome)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Feedback recorded: {'win' if outcome else 'loss'} for {signal_id}")

app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(CommandHandler("signal", signal_command))
app.add_handler(CommandHandler("won", lambda u, c: feedback_command(u, c, True)))
app.add_handler(CommandHandler("lost", lambda u, c: feedback_command(u, c, False)))
app.run_polling()