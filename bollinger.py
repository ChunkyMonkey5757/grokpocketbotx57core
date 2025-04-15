# bollinger.py
import pandas as pd
import logging
import indicator_base

logger = logging.getLogger('pocketbotx57.indicators.bollinger')

class BollingerBandsStrategy(indicator_base.IndicatorBase):
    def __init__(self):
        self.period = 20
        self.std_dev = 1.0  # Lowered from 1.5
        logger.info(f"BollingerBandsStrategy initialized: period={self.period}, std_dev={self.std_dev}")

    async def generate_signal(self, asset: str, data: pd.DataFrame):
        if len(data) < self.period:
            return {}

        rolling_mean = data['close'].rolling(window=self.period).mean()
        rolling_std = data['close'].rolling(window=self.period).std()
        upper_band = rolling_mean + (rolling_std * self.std_dev)
        lower_band = rolling_mean - (rolling_std * self.std_dev)
        latest_price = data['close'].iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]

        logger.info(f"Bollinger Bands for {asset}: Price={latest_price:.2f}, Upper={latest_upper:.2f}, Lower={latest_lower:.2f}")

        if latest_price < latest_lower:
            return {'action': 'BUY', 'confidence': 0.75, 'indicators': {'price': latest_price, 'upper': latest_upper, 'lower': latest_lower}, 'duration': 5}
        elif latest_price > latest_upper:
            return {'action': 'SELL', 'confidence': 0.75, 'indicators': {'price': latest_price, 'upper': latest_upper, 'lower': latest_lower}, 'duration': 5}
        return {}
