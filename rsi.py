"""
RSI Indicator Module for PocketBotX57
This module provides an enhanced RSI-based trading strategy.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union, Tuple
import uuid
from datetime import datetime
from indicator_base import IndicatorBase

# Setup logging
logger = logging.getLogger("pocketbotx57.indicators.rsi")

class RSIStrategy(IndicatorBase):
    """
    Enhanced RSI (Relative Strength Index) strategy.
    
    Features:
    - Standard RSI calculations with configurable periods
    - RSI divergence detection
    - Overbought/oversold levels with adaptive thresholds
    - RSI trend strength assessment
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize RSI strategy with configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # RSI parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.overbought_threshold = self.config.get('overbought_threshold', 70)
        self.oversold_threshold = self.config.get('oversold_threshold', 30)
        self.rsi_trend_period = self.config.get('rsi_trend_period', 5)
        self.divergence_lookback = self.config.get('divergence_lookback', 10)
        self.use_adaptive_thresholds = self.config.get('use_adaptive_thresholds', True)
        
        # Signal optimization parameters
        self.signal_cooldown = self.config.get('signal_cooldown', 5)  # periods
        self.min_confidence = self.config.get('min_confidence', 0.85)
        
        logger.info(f"RSI Strategy initialized with period={self.rsi_period}, "
                  f"thresholds={self.oversold_threshold}/{self.overbought_threshold}")
    
    async def calculate(self, market_data: pd.DataFrame) -> Dict:
        """
        Calculate RSI and related metrics.
        
        Args:
            market_data: DataFrame with OHLCV market data
            
        Returns:
            Dictionary with calculated values
        """
        if not self._validate_data(market_data):
            return {}
        
        # Calculate standard RSI
        close_prices = market_data['close']
        delta = close_prices.diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate RSI trend
        rsi_trend = rsi.diff(self.rsi_trend_period)
        
        # Detect RSI divergence
        divergence = self._detect_divergence(market_data, rsi)
        
        # Calculate adaptive thresholds if enabled
        if self.use_adaptive_thresholds:
            volatility = close_prices.pct_change().rolling(window=21).std().iloc[-1]
            
            # Adjust thresholds based on volatility
            adaptive_overbought = min(80, self.overbought_threshold + (volatility * 100))
            adaptive_oversold = max(20, self.oversold_threshold - (volatility * 100))
        else:
            adaptive_overbought = self.overbought_threshold
            adaptive_oversold = self.oversold_threshold
        
        result = {
            'rsi': rsi.iloc[-1],
            'rsi_values': rsi.values,
            'rsi_trend': rsi_trend.iloc[-1],
            'overbought_threshold': adaptive_overbought,
            'oversold_threshold': adaptive_oversold,
            'divergence': divergence
        }
        
        return result
    
    def _detect_divergence(self, market_data: pd.DataFrame, rsi: pd.Series) -> Dict:
        """
        Detect bullish and bearish RSI divergences.
        
        Args:
            market_data: OHLCV market data
            rsi: Calculated RSI series
            
        Returns:
            Dictionary with divergence information
        """
        # Use only the recent data for divergence detection
        lookback = min(self.divergence_lookback, len(market_data) - 1)
        
        prices = market_data['close'].iloc[-lookback:]
        rsi_values = rsi.iloc[-lookback:]
        
        # Find local price highs/lows and corresponding RSI values
        price_highs = []
        price_lows = []
        rsi_at_price_highs = []
        rsi_at_price_lows = []
        
        for i in range(1, len(prices) - 1):
            # Local price high
            if prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                price_highs.append((i, prices.iloc[i]))
                rsi_at_price_highs.append((i, rsi_values.iloc[i]))
            
            # Local price low
            if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                price_lows.append((i, prices.iloc[i]))
                rsi_at_price_lows.append((i, rsi_values.iloc[i]))
        
        # Need at least 2 highs/lows to detect divergence
        bullish_divergence = False
        bearish_divergence = False
        
        if len(price_lows) >= 2:
            # Check for bullish divergence (lower lows in price, higher lows in RSI)
            if (price_lows[-1][1] < price_lows[-2][1] and 
                rsi_at_price_lows[-1][1] > rsi_at_price_lows[-2][1]):
                bullish_divergence = True
        
        if len(price_highs) >= 2:
            # Check for bearish divergence (higher highs in price, lower highs in RSI)
            if (price_highs[-1][1] > price_highs[-2][1] and 
                rsi_at_price_highs[-1][1] < rsi_at_price_highs[-2][1]):
                bearish_divergence = True
        
        return {
            'bullish': bullish_divergence,
            'bearish': bearish_divergence
        }
    
    async def generate_signal(self, asset: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal based on RSI strategy.
        
        Args:
            asset: Asset symbol
            market_data: DataFrame with OHLCV market data
            
        Returns:
            Signal dictionary or None if no signal
        """
        # Calculate RSI values
        rsi_data = await self.calculate(market_data)
        
        if not rsi_data:
            return None
        
        # Extract values from calculation
        rsi_value = rsi_data['rsi']
        rsi_trend = rsi_data['rsi_trend']
        overbought = rsi_data['overbought_threshold']
        oversold = rsi_data['oversold_threshold']
        divergence = rsi_data['divergence']
        
        # Initialize signal
        signal = None
        confidence = 0.0
        
        # BUY signal conditions
        if rsi_value < oversold:
            # Oversold condition - potential BUY
            
            # Calculate base confidence from how oversold the asset is
            confidence = self._calculate_confidence(
                oversold - rsi_value,
                1,  # Min threshold
                15   # Max threshold
            )
            
            # Boost confidence if bullish divergence
            if divergence['bullish']:
                confidence = min(1.0, confidence + 0.15)
            
            # Boost confidence if RSI is turning up
            if rsi_trend > 0:
                confidence = min(1.0, confidence + 0.10)
            
            if confidence >= self.min_confidence:
                signal = {
                    'action': 'BUY',
                    'confidence': confidence,
                    'reason': f"RSI oversold ({rsi_value:.2f} < {oversold:.2f})"
                }
        
        # SELL signal conditions
        elif rsi_value > overbought:
            # Overbought condition - potential SELL
            
            # Calculate base confidence from how overbought the asset is
            confidence = self._calculate_confidence(
                rsi_value - overbought,
                1,  # Min threshold
                15   # Max threshold
            )
            
            # Boost confidence if bearish divergence
            if divergence['bearish']:
                confidence = min(1.0, confidence + 0.15)
            
            # Boost confidence if RSI is turning down
            if rsi_trend < 0:
                confidence = min(1.0, confidence + 0.10)
            
            if confidence >= self.min_confidence:
                signal = {
                    'action': 'SELL',
                    'confidence': confidence,
                    'reason': f"RSI overbought ({rsi_value:.2f} > {overbought:.2f})"
                }
        
        # If signal generated, add additional info
        if signal:
            # Add metadata for signal processing
            signal['id'] = str(uuid.uuid4())
            signal['timestamp'] = datetime.now().isoformat()
            signal['asset'] = asset
            signal['strategy'] = self.name
            
            # Determine optimal trading duration based on RSI extremeness
            extremeness = abs(rsi_value - 50) / 50  # 0 to 1 scale
            if extremeness > 0.7:
                duration = 1  # Very extreme RSI = short duration (quick reversal)
            elif extremeness > 0.4:
                duration = 3  # Moderate RSI extremeness = medium duration
            else:
                duration = 5  # Mild RSI signal = longer duration
            
            signal['duration'] = duration
            
            # Add indicators data for analysis
            signal['indicators'] = {
                'rsi': rsi_value,
                'rsi_trend': rsi_trend,
                'divergence': divergence
            }
            
            logger.info(f"RSI signal generated for {asset}: {signal['action']} with "
                       f"{signal['confidence']:.2%} confidence, duration: {duration}m")
        
        return signal