import unittest
from collections import namedtuple
import logging
from package.hand_raise_detection import HandRaiseDetector

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


class TestHandRaiseDetection(unittest.TestCase):
    def setUp(self):
        self.detector = HandRaiseDetector()
        self.Landmark = namedtuple("Landmark", ["x", "y", "z", "visibility"])

    def create_landmarks(
        self, left_wrist_y, right_wrist_y, left_shoulder_y=0.5, right_shoulder_y=0.5
    ):
        """
        Create a set of landmarks for testing hand raise detection.

        Parameters:
            left_wrist_y (float): Y-coordinate for left wrist.
            right_wrist_y (float): Y-coordinate for right wrist.
            left_shoulder_y (float): Y-coordinate for left shoulder (default 0.5).
            right_shoulder_y (float): Y-coordinate for right shoulder (default 0.5).

        Returns:
            list: List of named landmarks with their respective Y-coordinates.
        """
        landmarks = [self.Landmark(0, 0, 0, 0)] * 33  # 33 landmarks in MediaPipe

        def set_landmark(index, y_val):
            landmarks[index] = self.Landmark(0, y_val, 0, 0)

        set_landmark(
            self.detector.mp_pose.PoseLandmark.LEFT_SHOULDER.value, left_shoulder_y
        )
        set_landmark(
            self.detector.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, right_shoulder_y
        )
        set_landmark(self.detector.mp_pose.PoseLandmark.LEFT_WRIST.value, left_wrist_y)
        set_landmark(
            self.detector.mp_pose.PoseLandmark.RIGHT_WRIST.value, right_wrist_y
        )

        return landmarks

    def test_left_hand_raised(self):
        landmarks = self.create_landmarks(left_wrist_y=0.3, right_wrist_y=0.6)
        logging.info("Running test: Left hand raised.")
        self.assertTrue(self.detector.is_hand_raised(landmarks))

    def test_right_hand_raised(self):
        landmarks = self.create_landmarks(left_wrist_y=0.6, right_wrist_y=0.3)
        logging.info("Running test: Right hand raised.")
        self.assertTrue(self.detector.is_hand_raised(landmarks))

    def test_both_hands_raised(self):
        landmarks = self.create_landmarks(left_wrist_y=0.2, right_wrist_y=0.2)
        logging.info("Running test: Both hands raised.")
        self.assertTrue(self.detector.is_hand_raised(landmarks))

    def test_no_hands_raised(self):
        landmarks = self.create_landmarks(left_wrist_y=0.6, right_wrist_y=0.6)
        logging.info("Running test: No hands raised.")
        self.assertFalse(self.detector.is_hand_raised(landmarks))


if __name__ == "__main__":
    unittest.main()
