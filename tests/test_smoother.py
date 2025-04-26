import unittest
from package.smoother import RealTimeSmoother


class TestRealTimeSmoother(unittest.TestCase):
    """Unit tests for the RealTimeSmoother class."""

    def test_average(self):
        """Test moving average computation with incremental updates."""
        smoother = RealTimeSmoother(window_size=5)

        smoother.update(10.0)
        avg: float = smoother.get_average()
        self.assertAlmostEqual(avg, 10.0, delta=0.01)

        smoother.update(12.5)
        avg = smoother.get_average()
        self.assertAlmostEqual(avg, 11.25, delta=0.01)

        smoother.update(12.5)
        avg = smoother.get_average()
        self.assertAlmostEqual(avg, 11.6666666667, delta=0.01)


if __name__ == "__main__":
    unittest.main()
