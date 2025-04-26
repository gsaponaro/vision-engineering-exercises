"""
Exercise 1: Moving Average Smoother for Real-Time Sensor Data

This module defines a RealTimeSmoother class that maintains a moving average
of the last N values from a sensor stream.

Example usage:
    smoother = RealTimeSmoother(window_size=5)
    smoother.update(10.0)
    smoother.update(12.5)
    current_avg = smoother.get_average()

Handles edge cases such as:
    - No values yet (returns 0.0)
    - Outliers are naturally smoothed by the moving average

Topics: State management and filtering
"""

from collections import deque  # Actual class where we store values
from typing import Deque  # Just for type hint annotations


class RealTimeSmoother:
    """Maintains a moving average over the last N values."""

    def __init__(self, window_size: int = 5):
        """
        Initialize the smoother.

        Args:
            window_size (int): Number of recent values to average over.
        """
        self.window_size: int = window_size
        self.values: Deque[float] = deque(maxlen=window_size)

    def update(self, new_value: float) -> None:
        """
        Add a new value to the window.

        Args:
            new_value (float): The new sensor reading to include.
        """
        self.values.append(new_value)

    def get_average(self) -> float:
        """
        Calculate the current moving average.

        Returns:
            float: The average of the most recent values,
                   or 0.0 if no values have been added.
        """
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
