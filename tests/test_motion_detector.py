import unittest
from package.motion_detector import detect_motion
import cv2
import numpy as np


class TestMotionDetection(unittest.TestCase):
    """
    Unit tests for the motion detection functionality.
    These tests cover scenarios with no motion, small motion below threshold,
    clear motion, and multiple motion regions.
    """

    def setUp(self):
        """
        Sets up a blank frame for use in tests.
        """
        # Dimensions for test frames
        self.height, self.width = 240, 320

        # Create a blank frame (black image)
        self.blank_frame = np.zeros((self.height, self.width), dtype=np.uint8)

    def test_no_motion(self):
        """
        Test case where no motion occurs between two identical frames.
        """
        regions = detect_motion(self.blank_frame, self.blank_frame)
        self.assertEqual(
            len(regions), 0, "No motion should be detected for identical frames."
        )

    def test_small_motion_below_threshold(self):
        """
        Test case where motion is detected, but it's too small to be considered valid.
        """
        frame2 = self.blank_frame.copy()
        # Create a small rectangle (too small to be detected)
        cv2.rectangle(frame2, (10, 10), (12, 12), 255, -1)

        regions = detect_motion(self.blank_frame, frame2)
        self.assertEqual(
            len(regions), 0, "Small motion below MIN_AREA should not be detected."
        )

    def test_clear_motion(self):
        """
        Test case where a large enough region of motion is detected.
        """
        frame2 = self.blank_frame.copy()
        # Create a large rectangle that should trigger motion detection
        cv2.rectangle(frame2, (50, 50), (80, 80), 255, -1)

        regions = detect_motion(self.blank_frame, frame2)
        self.assertGreaterEqual(
            len(regions),
            1,
            "Motion should be detected for a sufficiently large region.",
        )

    def test_multiple_regions(self):
        """
        Test case where multiple regions of motion are detected.
        """
        frame2 = self.blank_frame.copy()
        # Create two separate rectangles, both large enough to be detected
        cv2.rectangle(frame2, (10, 10), (40, 40), 255, -1)
        cv2.rectangle(frame2, (100, 100), (130, 130), 255, -1)

        regions = detect_motion(self.blank_frame, frame2)
        self.assertEqual(
            len(regions), 2, "Two distinct motion regions should be detected."
        )


if __name__ == "__main__":
    unittest.main()
