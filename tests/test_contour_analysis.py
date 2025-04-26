import unittest
import cv2
import numpy as np
from package.contour_analysis import analyze_contours, draw_contours_and_bboxes


class TestContourAnalysis(unittest.TestCase):

    def setUp(self):
        # Create a simple black image and a white rectangle (representing a contour)
        self.image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(self.image, (50, 50), (150, 150), 255, -1)  # Draw white rectangle

    def test_analyze_contours(self):
        # Call the analyze_contours function
        contours_info = analyze_contours(self.image, area_threshold=100)

        # Test that we get one contour (the rectangle we drew)
        self.assertEqual(len(contours_info), 1)

        # Check bounding box (x, y, width, height)
        bbox = contours_info[0]["bbox"]
        self.assertEqual(bbox, (50, 50, 101, 101))  # Rectangle's bounding box

        # Check contour area
        area = contours_info[0]["area"]
        self.assertEqual(area, 10000.0)  # Area of the rectangle (101 * 101)

        # Check the number of corners (should be 4 for a rectangle)
        corners = contours_info[0]["corners"]
        self.assertEqual(corners, 4)  # Rectangle has 4 corners

    def test_analyze_contours_area_threshold(self):
        # Create a new image with a smaller contour (below threshold)
        small_image = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(small_image, (100, 100), 5, 255, -1)  # Draw small circle

        # Analyze contours with threshold higher than the small contour's area
        contours_info = analyze_contours(small_image, area_threshold=100)

        # Assert that no contours are detected since the area is smaller than the threshold
        self.assertEqual(len(contours_info), 0)

    def test_draw_contours_and_bboxes(self):
        # Create an image with a single white rectangle
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)

        # Analyze the contours
        contours_info = analyze_contours(image, area_threshold=100)

        # Draw the contours and bounding boxes on the image
        result_image = draw_contours_and_bboxes(image.copy(), contours_info)

        # Ensure the resulting image has been modified (i.e., the rectangle is drawn)
        self.assertTrue(np.any(result_image != image))

    def test_empty_image(self):
        # Test an empty image (no contours)
        empty_image = np.zeros((200, 200), dtype=np.uint8)

        # Analyze contours on the empty image
        contours_info = analyze_contours(empty_image, area_threshold=100)

        # Assert that no contours are found
        self.assertEqual(len(contours_info), 0)


if __name__ == "__main__":
    unittest.main()
