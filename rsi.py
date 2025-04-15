# rsi.py
import pandas as pd
import logging
import indicator_base

logger = logging.getLogger('pocketbotx57.indicators.rsi')

class RSIStrategy(indicator_base.IndicatorBase):
    def __init__(self):
        self.period = 14
        self.overbought = 65  # Lowered from 70
        self.oversold = 35    # Raised from 30
        logger.info(f"RSI Strategy initialized with period={self.period}, thresholds={self.oversold}/{self.overbought}")

    async def generate_signal(self, asset: str, data: pd.DataFrame):
        if len(data) < self.period:
            return {}

        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        latest_rsi = rsi.iloc[-1]

        if pd.isna(latest_rsi):
            logger.warning(f"RSI calculation resulted in NaN for {asset}")
            return {}

        logger.info(f"RSI for {asset}: {latest_rsi:.2f}")

        if latest_rsi < self.oversold:
            return {'action': 'BUY', 'confidence': 0.8, 'indicators': {'rsi': latest_rsi}, 'duration': 5}
        elif latest_rsi > self.overbought:
            return {'action': 'SELL', 'confidence': 0.8, 'indicators': {'rsi': latest_rsi}, 'duration': 5}
        return {}
