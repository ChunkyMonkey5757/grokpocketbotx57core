from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import logging

logger = logging.getLogger("pocketbotx57.indicators.base")

class IndicatorBase(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.config.get("name", "indicator")
        self.min_confidence = self.config.get("min_confidence", 0.85)
        logger.info(f"IndicatorBase initialized for {self.name}")

    @abstractmethod
    async def generate_signal(self, asset: str, data: pd.DataFrame) -> Optional[Dict]:
        pass

    def _validate_data(self, data: pd.DataFrame, required_columns: List[str] = None) -> bool:
        if required_columns is None:
            required_columns = ["close"]

        if data is None or data.empty:
            logger.warning("Empty dataset provided")
            return False

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False

        if len(data) < 50:
            logger.warning(f"Insufficient data points: {len(data)} < 50")
            return False

        return True

    def _calculate_confidence(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return 0.5
        confidence = 0.5 + (value - min_val) / (max_val - min_val)
        return min(1.0, max(0.5, confidence))