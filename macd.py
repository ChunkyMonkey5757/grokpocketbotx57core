"""
MACD Indicator Module for PocketBotX57
This module provides an enhanced MACD-based trading strategy.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, Tuple
import uuid
from datetime import datetime


from indicator_base import IndicatorBase

# Setup logging
logger = logging.getLogger("pocketbotx57.indicators.macd")

class MACDStrategy(IndicatorBase):
    """
    Enhanced MACD (Moving Average Convergence Divergence) strategy.
    
    Features:
    - Standard MACD calculations with configurable periods
    - Signal line crossover detection
    - Histogram trend analysis
    - Zero-line crossovers
    - MACD divergence detection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize MACD strategy with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # MACD parameters
        self.fast_period = self.config.get('fast_period', 12)
        self.slow_period = self.config.get('slow_period', 26)
        self.signal_period = self.config.get('signal_period', 9)
        self.divergence_lookback = self.config.get('divergence_lookback', 10)
        
        # Signal optimization parameters
        self.signal_cooldown = self.config.get('signal_cooldown', 5)  # periods
        self.min_confidence = self.config.get('min_confidence', 0.85)
        
        logger.info(f"MACD Strategy initialized with periods={self.fast_period}/{self.slow_period}/{self.signal_period}")
    
    async def calculate(self, market_data: pd.DataFrame) -> Dict:
        """
        Calculate MACD and related metrics.
        
        Args:
            market_data: DataFrame with OHLCV market data
            
        Returns:
            Dictionary with calculated values
        """
        if not self._validate_data(market_data):
            return {}
        
        # Calculate MACD components
        close_prices = market_data['close']
        
        # Calculate EMAs
        ema_fast = close_prices.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close_prices.ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Detect crossovers
        crossovers = self._detect_crossovers(macd_line, signal_line)
        
        # Detect zero-line crossovers
        zero_crossovers = self._detect_zero_crossovers(macd_line)
        
        # Calculate histogram trend
        histogram_trend = histogram.diff(3)
        
        # Detect MACD divergence
        divergence = self._detect_divergence(market_data, macd_line)
        
        # Check for histogram reversal pattern
        histogram_reversal = self._detect_histogram_reversal(histogram)
        
        result = {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1],
            'histogram_trend': histogram_trend.iloc[-1],
            'crossover': crossovers,
            'zero_crossover': zero_crossovers,
            'divergence': divergence,
            'histogram_reversal': histogram_reversal,
            'macd_values': macd_line.values,
            'signal_values': signal_line.values,
            'histogram_values': histogram.values
        }
        
        return result
    
    def _detect_crossovers(self, macd_line: pd.Series, signal_line: pd.Series) -> Dict:
        """
        Detect MACD and signal line crossovers.
        
        Args:
            macd_line: MACD line series
            signal_line: Signal line series
            
        Returns:
            Dictionary with crossover information
        """
        # Need at least 2 periods to detect crossovers
        if len(macd_line) < 2 or len(signal_line) < 2:
            return {'bullish': False, 'bearish': False}
        
        # Check for bullish crossover (MACD crosses above signal)
        bullish = (macd_line.iloc[-2] < signal_line.iloc[-2] and 
                  macd_line.iloc[-1] > signal_line.iloc[-1])
        
        # Check for bearish crossover (MACD crosses below signal)
        bearish = (macd_line.iloc[-2] > signal_line.iloc[-2] and 
                  macd_line.iloc[-1] < signal_line.iloc[-1])
        
        return {
            'bullish': bullish,
            'bearish': bearish
        }
    
    def _detect_zero_crossovers(self, macd_line: pd.Series) -> Dict:
        """
        Detect MACD zero-line crossovers.
        
        Args:
            macd_line: MACD line series
            
        Returns:
            Dictionary with zero-line crossover information
        """
        # Need at least 2 periods to detect crossovers
        if len(macd_line) < 2:
            return {'bullish': False, 'bearish': False}
        
        # Check for bullish zero-line crossover (MACD crosses above zero)
        bullish = macd_line.iloc[-2] < 0 and macd_line.iloc[-1] > 0
        
        # Check for bearish zero-line crossover (MACD crosses below zero)
        bearish = macd_line.iloc[-2] > 0 and macd_line.iloc[-1] < 0
        
        return {
            'bullish': bullish,
            'bearish': bearish
        }
    
    def _detect_divergence(self, market_data: pd.DataFrame, macd: pd.Series) -> Dict:
        """
        Detect bullish and bearish MACD divergences.
        
        Args:
            market_data: OHLCV market data
            macd: MACD line series
            
        Returns:
            Dictionary with divergence information
        """
        # Use only the recent data for divergence detection
        lookback = min(self.divergence_lookback, len(market_data) - 1)
        
        prices = market_data['close'].iloc[-lookback:]
        macd_values = macd.iloc[-lookback:]
        
        # Find local price highs/lows and corresponding MACD values
        price_highs = []
        price_lows = []
        macd_at_price_highs = []
        macd_at_price_lows = []
        
        for i in range(1, len(prices) - 1):
            # Local price high
            if prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                price_highs.append((i, prices.iloc[i]))
                macd_at_price_highs.append((i, macd_values.iloc[i]))
            
            # Local price low
            if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                price_lows.append((i, prices.iloc[i]))
                macd_at_price_lows.append((i, macd_values.iloc[i]))
        
        # Need at least 2 highs/lows to detect divergence
        bullish_divergence = False
        bearish_divergence = False
        
        if len(price_lows) >= 2:
            # Check for bullish divergence (lower lows in price, higher lows in MACD)
            if (price_lows[-1][1] < price_lows[-2][1] and 
                macd_at_price_lows[-1][1] > macd_at_price_lows[-2][1]):
                bullish_divergence = True
        
        if len(price_highs) >= 2:
            # Check for bearish divergence (higher highs in price, lower highs in MACD)
            if (price_highs[-1][1] > price_highs[-2][1] and 
                macd_at_price_highs[-1][1] < macd_at_price_highs[-2][1]):
                bearish_divergence = True
        
        return {
            'bullish': bullish_divergence,
            'bearish': bearish_divergence
        }
    
    def _detect_histogram_reversal(self, histogram: pd.Series) -> Dict:
        """
        Detect histogram reversal patterns.
        
        Args:
            histogram: MACD histogram series
            
        Returns:
            Dictionary with histogram reversal information
        """
        # Need at least 4 periods to detect reversal pattern
        if len(histogram) < 4:
            return {'bullish': False, 'bearish': False}
        
        # Get last 4 histogram values
        h4 = histogram.iloc[-4]
        h3 = histogram.iloc[-3]
        h2 = histogram.iloc[-2]
        h1 = histogram.iloc[-1]
        
        # Bullish reversal: Decreasing negative values then increasing
        bullish = h4 < h3 < h2 < 0 and h1 > h2
        
        # Bearish reversal: Increasing positive values then decreasing
        bearish = h4 > h3 > h2 > 0 and h1 < h2
        
        return {
            'bullish': bullish,
            'bearish': bearish
        }
    
    async def generate_signal(self, asset: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal based on MACD strategy.
        
        Args:
            asset: Asset symbol
            market_data: DataFrame with OHLCV market data
            
        Returns:
            Signal dictionary or None if no signal
        """
        # Calculate MACD values
        macd_data = await self.calculate(market_data)
        
        if not macd_data:
            return None
        
        # Extract values from calculation
        macd_value = macd_data['macd']
        signal_value = macd_data['signal']
        histogram = macd_data['histogram']
        histogram_trend = macd_data['histogram_trend']
        crossover = macd_data['crossover']
        zero_crossover = macd_data['zero_crossover']
        divergence = macd_data['divergence']
        histogram_reversal = macd_data['histogram_reversal']
        
        # Initialize signal
        signal = None
        confidence = 0.0
        
        # BUY signal conditions
        if crossover['bullish'] or zero_crossover['bullish'] or histogram_reversal['bullish']:
            # Base confidence from crossover type
            if crossover['bullish']:
                confidence = 0.7
                reason = "Bullish MACD crossover"
            elif zero_crossover['bullish']:
                confidence = 0.75
                reason = "Bullish MACD zero-line crossover"
            elif histogram_reversal['bullish']:
                confidence = 0.65
                reason = "Bullish MACD histogram reversal"
            
            # Boost confidence if multiple conditions align
            if histogram > 0:
                confidence = min(1.0, confidence + 0.05)
            
            if histogram_trend > 0:
                confidence = min(1.0, confidence + 0.05)
            
            if divergence['bullish']:
                confidence = min(1.0, confidence + 0.1)
            
            if zero_crossover['bullish'] and crossover['bullish']:
                confidence = min(1.0, confidence + 0.15)
            
            if confidence >= self.min_confidence:
                signal = {
                    'action': 'BUY',
                    'confidence': confidence,
                    'reason': reason
                }
        
        # SELL signal conditions
        elif crossover['bearish'] or zero_crossover['bearish'] or histogram_reversal['bearish']:
            # Base confidence from crossover type
            if crossover['bearish']:
                confidence = 0.7
                reason = "Bearish MACD crossover"
            elif zero_crossover['bearish']:
                confidence = 0.75
                reason = "Bearish MACD zero-line crossover"
            elif histogram_reversal['bearish']:
                confidence = 0.65
                reason = "Bearish MACD histogram reversal"
            
            # Boost confidence if multiple conditions align
            if histogram < 0:
                confidence = min(1.0, confidence + 0.05)
            
            if histogram_trend < 0:
                confidence = min(1.0, confidence + 0.05)
            
            if divergence['bearish']:
                confidence = min(1.0, confidence + 0.1)
            
            if zero_crossover['bearish'] and crossover['bearish']:
                confidence = min(1.0, confidence + 0.15)
            
            if confidence >= self.min_confidence:
                signal = {
                    'action': 'SELL',
                    'confidence': confidence,
                    'reason': reason
                }
        
        # If signal generated, add additional info
        if signal:
            # Add metadata for signal processing
            signal['id'] = str(uuid.uuid4())
            signal['timestamp'] = datetime.now().isoformat()
            signal['asset'] = asset
            signal['strategy'] = self.name
            
            # Determine optimal trading duration
            # MACD is a trend indicator, so signals often work better with longer durations
            signal['duration'] = 5  # Default to 5 minutes
            
            # Add indicators data for analysis
            signal['indicators'] = {
                'macd': macd_value,
                'macd_signal': signal_value,
                'histogram': histogram,
                'divergence': divergence
            }
            
            logger.info(f"MACD signal generated for {asset}: {signal['action']} with "
                       f"{signal['confidence']:.2%} confidence")
        
        return signal