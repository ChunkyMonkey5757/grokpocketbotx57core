import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from rsi import RSIStrategy
from macd import MACDStrategy
from bollinger import BollingerBandsStrategy

logger = logging.getLogger("pocketbotx57.signal_engine")

class SignalEngine:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.indicators = {
            'rsi': RSIStrategy(),
            'macd': MACDStrategy(),
            'bollinger': BollingerBandsStrategy()
        }
        self.weights = {'rsi': 0.15, 'macd': 0.20, 'bollinger': 0.15}
        self.min_confidence = self.config.get('min_confidence', 0.85)
        self.cooldown_period = self.config.get('cooldown_period', 120)
        self.last_signal_time = {}
        self.signal_history = []
        self.learning_rate = 0.05
        logger.info("Signal Engine (MVP + AI Lite) initialized")

    async def process_market_data(self, asset: str, data: pd.DataFrame) -> Optional[Dict]:
        current_time = datetime.now().timestamp()
        if asset in self.last_signal_time and (current_time - self.last_signal_time[asset]) < self.cooldown_period:
            return None

        if len(data) < 50:
            logger.warning(f"Insufficient data for {asset}: {len(data)}")
            return None

        signals = await self._generate_indicator_signals(asset, data)
        signal = self._combine_indicator_signals(signals)
        
        if signal and signal['confidence'] >= self.min_confidence:
            signal['id'] = str(np.random.randint(10000, 99999))
            signal['asset'] = asset
            signal['timestamp'] = datetime.now().isoformat()
            self.last_signal_time[asset] = current_time
            self.signal_history.append(signal)
            logger.info(f"Signal for {asset}: {signal['action']} ({signal['confidence']:.2%})")
            return signal
        return None

    async def _generate_indicator_signals(self, asset: str, data: pd.DataFrame) -> Dict:
        tasks = [self._run_indicator(name, ind, asset, data) for name, ind in self.indicators.items()]
        return dict(await asyncio.gather(*tasks))

    async def _run_indicator(self, name: str, indicator, asset: str, data: pd.DataFrame):
        try:
            signal = await indicator.generate_signal(asset, data)
            return (name, signal or {})
        except Exception as e:
            logger.error(f"Indicator {name} failed: {str(e)}")
            return (name, {})

    def _combine_indicator_signals(self, signals: Dict) -> Optional[Dict]:
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
            return None

        return {
            'action': action,
            'confidence': confidence,
            'duration': max([s.get('duration', 5) for s in valid_signals.values()]),
            'indicators': {k: v['indicators'] for k, v in valid_signals.items()},
            'contributing_strategies': list(valid_signals.keys())
        }

    async def process_feedback(self, signal_id: str, outcome: bool):
        signal = next((s for s in self.signal_history if str(s['id']) == signal_id), None)
        if not signal:
            logger.warning(f"Signal {signal_id} not found")
            return

        signal['outcome'] = outcome
        for strategy in signal.get('contributing_strategies', []):
            if outcome and signal['action'] == signals[strategy]['action']:
                self.weights[strategy] = min(0.35, self.weights[strategy] + self.learning_rate)
            elif not outcome:
                self.weights[strategy] = max(0.05, self.weights[strategy] - self.learning_rate)
        
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total / 0.5
        logger.info(f"Feedback processed for {signal_id}: {'win' if outcome else 'loss'}, weights: {self.weights}")

    def format_signal_message(self, signal: Dict, price: float) -> str:
        emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
        return (
            f"{emoji} *{signal['action']} Signal*\n"
            f"Asset: {signal['asset']}\n"
            f"Price: ${price:.2f}\n"
            f"Confidence: {signal['confidence']:.1%}\n"
            f"Duration: {signal['duration']}m\n"
            f"Signal ID: {signal['id']}\n"
            f"*TRADE NOW!*"
        )