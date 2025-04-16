import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import uuid

import rsi
import macd
import bollinger

logger = logging.getLogger('pocketbotx57.signal_engine')

class SignalEngine:
    def __init__(self, config: Dict = None):
        """Initialize the SignalEngine."""
        self.config = config or {}
        self.indicators = {
            'rsi': rsi.RSIStrategy(),
            'macd': macd.MACDStrategy(),
            'bollinger': bollinger.BollingerBandsStrategy()
        }
        self.weights = {'rsi': 0.3, 'macd': 0.4, 'bollinger': 0.3}
        self.min_confidence = 0.1
        self.cooldown_period = self.config.get('cooldown_period', 30)
        self.last_signal_time = {}
        self.signal_history = []
        self.learning_rate = 0.05
        logger.info("Signal Engine (MVP + AI Lite) initialized")

    async def process_market_data(self, asset: str, data: pd.DataFrame) -> Optional[Dict]:
        """Process market data to generate a trading signal."""
        current_time = datetime.now().timestamp()
        if asset in self.last_signal_time and (current_time - self.last_signal_time[asset]) < self.cooldown_period:
            return None

        if len(data) < 50:
            logger.warning(f"Insufficient data for {asset}: {len(data)}")
            return None

        signals = await self._generate_indicator_signals(asset, data)
        signal = self._combine_indicator_signals(signals)

        if signal:
            signal['asset'] = asset
            signal['timestamp'] = datetime.now().isoformat()
            signal['id'] = str(uuid.uuid4())  # Unique ID for each signal
            signal['start_time'] = datetime.now().isoformat()  # Added for format_signal_message
            signal['trade_duration'] = 15  # Default trade duration in minutes (adjust as needed)
            self.last_signal_time[asset] = current_time
            self.signal_history.append(signal)
            logger.info(f"Signal for {asset}: {signal['action']} ({signal['confidence']:.2%})")
            return signal
        return None

    async def _generate_indicator_signals(self, asset: str, data: pd.DataFrame) -> Dict:
        """Generate signals from all indicators asynchronously."""
        tasks = [self._run_indicator(name, ind, asset, data) for name, ind in self.indicators.items()]
        results = dict(await asyncio.gather(*tasks))
        logger.info(f"Indicator signals for {asset}: {results}")
        return results

    async def _run_indicator(self, name: str, indicator, asset: str, data: pd.DataFrame):
        """Run a single indicator and handle exceptions."""
        try:
            signal = await indicator.generate_signal(asset, data)
            return (name, signal or {})
        except Exception as e:
            logger.error(f"Indicator {name} failed: {str(e)}")
            return (name, {})

    def _combine_indicator_signals(self, signals: Dict) -> Optional[Dict]:
        """Combine signals from indicators into a single trading signal."""
        valid_signals = {k: v for k, v in signals.items() if v and 'action' in v}
        if not valid_signals:
            return None

        buy_score = sum(self.weights[k] * v['confidence'] for k, v in valid_signals.items() if v['action'] == 'BUY')
        sell_score = sum(self.weights[k] * v['confidence'] for k, v in valid_signals.items() if v['action'] == 'SELL')

        if buy_score > sell_score:
            action, confidence = 'BUY', buy_score
        elif sell_score > buy_score:
            action, confidence = 'SELL', sell_score
        else:
            max_confidence_signal = max(valid_signals.items(), key=lambda x: x[1]['confidence'], default=(None, {}))
            if max_confidence_signal[0] is None:
                return None
            action = max_confidence_signal[1]['action']
            confidence = max_confidence_signal[1]['confidence']
            valid_signals = {max_confidence_signal[0]: max_confidence_signal[1]}

        num_contributors = len(valid_signals)
        if num_contributors > 0:
            confidence = confidence * (1.0 / sum(self.weights[k] for k in valid_signals))

        return {
            'action': action,
            'confidence': confidence,
            'duration': max(v.get('duration', 5) for v in valid_signals.values()),
            'indicators': {k: v['indicators'] for k, v in valid_signals.items()},
            'contributing_strategies': list(valid_signals.keys())
        }

    async def process_feedback(self, signal_id: str, outcome: bool):
        """Process feedback to adjust strategy weights."""
        signal = next((s for s in self.signal_history if str(s['id']) == signal_id), None)
        if not signal:
            logger.warning(f"Signal {signal_id} not found")
            return

        signal['outcome'] = outcome
        for strategy in signal.get('contributing_strategies', []):
            if outcome:
                self.weights[strategy] = min(0.35, self.weights[strategy] + self.learning_rate)
            else:
                self.weights[strategy] = max(0.05, self.weights[strategy] - self.learning_rate)

        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] = total / 1.0

        logger.info(f"Feedback processed for {signal_id}: {'win' if outcome else 'loss'}, weights: {self.weights}")

    def format_signal_message(self, signal: Dict) -> str:
        """Format a signal into a Telegram-compatible message."""
        emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
        logic_explanation = self._generate_logic_explanation(signal)
        return (
            f"{emoji} **{signal['action']} Signal for {signal['asset']}**\n"
            f"**Confidence:** {signal['confidence']:.1%}\n"
            f"**Begin Trade At:** {signal.get('start_time', 'N/A')}\n"
            f"**Entry Window:** {signal['duration']} minutes\n"
            f"**Trade Duration:** {signal.get('trade_duration', 'N/A')} minutes\n"
            f"**Logic:** {logic_explanation}\n"
            f"**TRADE NOW!**"
        )

    def _generate_logic_explanation(self, signal: Dict) -> str:
        """Generate a human-readable explanation of the signal logic."""
        contributing = signal['contributing_strategies']
        indicators = signal['indicators']
        explanations = []
        for strategy in contributing:
            if strategy == 'rsi':
                rsi_value = indicators[strategy]['rsi']
                explanations.append(f"RSI ({rsi_value:.2f}) indicates {'oversold' if signal['action'] == 'BUY' else 'overbought'} conditions")
            elif strategy == 'macd':
                macd_value = indicators[strategy]['macd']
                signal_value = indicators[strategy]['signal']
                explanations.append(f"MACD ({macd_value:.2f}) crossed {'above' if signal['action'] == 'BUY' else 'below'} signal line ({signal_value:.2f})")
            elif strategy == 'bollinger':
                price = indicators[strategy]['price']
                upper = indicators[strategy]['upper']
                lower = indicators[strategy]['lower']
                explanations.append(f"Price ({price:.2f}) crossed {'below lower band' if signal['action'] == 'BUY' else 'above upper band'} (Upper: {upper:.2f}, Lower: {lower:.2f})")
        return "; ".join(explanations) if explanations else "Based on combined indicator analysis"