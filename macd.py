# macd.py
import pandas as pd
import logging
import indicator_base  # Changed from relative to absolute import

logger = logging.getLogger('pocketbotx57.indicators.macd')

class MACDStrategy(indicator_base.IndicatorBase):
    def __init__(self):
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        logger.info(f"MACD Strategy initialized with periods={self.fast_period}/{self.slow_period}/{self.signal_period}")

    async def generate_signal(self, asset: str, data: pd.DataFrame):
        if len(data) < self.slow_period:
            return {}

        exp1 = data['close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = data['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]

        logger.info(f"MACD for {asset}: MACD={latest_macd:.4f}, Signal={latest_signal:.4f}")

        if latest_macd > latest_signal:
            return {'action': 'BUY', 'confidence': 0.7, 'indicators': {'macd': latest_macd, 'signal': latest_signal}, 'duration': 5}
        elif latest_macd < latest_signal:
            return {'action': 'SELL', 'confidence': 0.7, 'indicators': {'macd': latest_macd, 'signal': latest_signal}, 'duration': 5}
        return {}