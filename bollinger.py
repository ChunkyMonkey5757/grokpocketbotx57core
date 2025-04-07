"""
Bollinger Bands Strategy for PocketBotX57 (MVP + AI Lite)
Simple, fast Bollinger Bands for signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from pandas_ta import bbands
from indicator_base import IndicatorBase

logger = logging.getLogger("pocketbotx57.indicators.bollinger")

class BollingerBandsStrategy(IndicatorBase):
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.name = "bollinger"
        self.period = self.config.get('period', 20)
        self.std_dev = self.config.get('std_dev', 2.0)
        logger.info(f"BollingerBandsStrategy initialized: period={self.period}, std_dev={self.std_dev}")

    async def generate_signal(self, asset: str, data: pd.DataFrame) -> Optional[Dict]:
        """Generate async trading signal for SignalEngine."""
        if not self._validate_data(data):
            return None

        # Use pandas_ta for fast Bollinger Bands
        bb = bbands(data['close'], length=self.period, std=self.std_dev)
        if bb is None or bb.empty:
            return None

        upper = bb[f'BBU_{self.period}_{self.std_dev}'].iloc[-1]
        middle = bb[f'BBM_{self.period}_{self.std_dev}'].iloc[-1]
        lower = bb[f'BBL_{self.period}_{self.std_dev}'].iloc[-1]
        current_close = data['close'].iloc[-1]

        signal = None
        if current_close <= lower:
            confidence = self._calculate_confidence(
                lower - current_close,
                0,
                lower * 0.02  # Scale confidence based on 2% below lower band
            )
            signal = {
                'action': 'BUY',
                'confidence': confidence,
                'duration': 5,
                'indicators': {'upper': upper, 'middle': middle, 'lower': lower}
            }
        elif current_close >= upper:
            confidence = self._calculate_confidence(
                current_close - upper,
                0,
                upper * 0.02  # Scale confidence based on 2% above upper band
            )
            signal = {
                'action': 'SELL',
                'confidence': confidence,
                'duration': 5,
                'indicators': {'upper': upper, 'middle': middle, 'lower': lower}
            }

        if signal and signal['confidence'] >= self.min_confidence:
            logger.info(f"Bollinger signal for {asset}: {signal['action']} ({signal['confidence']:.2f})")
            return signal
        return None