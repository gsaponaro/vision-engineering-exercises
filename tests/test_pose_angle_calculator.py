import unittest
from package.pose_angle_calculator import calculate_angle


class TestPoseAngleCalculator(unittest.TestCase):
    def test_elbow_angle(self):
        """
        Test that the calculate_angle function computes the elbow angle correctly.

        This test uses a set of 2D keypoints representing the shoulder, elbow, and wrist.
        It verifies that the calculated angle is within a reasonable range (0 to 180 degrees)
        and asserts that it is approximately equal to the expected value.
        """

        keypoints = {
            "shoulder": (4.0, 5.0),
            "elbow": (52.4, 42.42),
            "wrist": (12.0, 14.0),
        }

        # Call the function to calculate the angle
        angle = calculate_angle(keypoints)

        # Print the result for debugging purposes (optional)
        print(f"Elbow angle: {angle:.2f} degrees")

        # Assert that the angle is within a reasonable range (0 to 180 degrees)
        self.assertGreaterEqual(angle, 0)
        self.assertLessEqual(angle, 180)

        # Optionally assert that the calculated angle is approximately equal to the expected value
        self.assertAlmostEqual(angle, 2.58, places=2)


if __name__ == "__main__":
    unittest.main()
