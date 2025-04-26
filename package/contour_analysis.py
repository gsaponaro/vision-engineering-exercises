"""
Exercise 4: Contour Analysis

Given a binary image (e.g., from thresholding), extract contours and compute:
- Bounding box
- Contour area
- Shape approximation (e.g., number of corners)

Add filtering so only contours with area > threshold are considered.

Topics: Shape analysis
"""

import cv2
import numpy as np
import os

# Suppress the error message
# qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
os.environ["QT_QPA_PLATFORM"] = "xcb"

AREA_THRESHOLD: int = 100


def analyze_contours(image: np.ndarray, area_threshold: int = AREA_THRESHOLD) -> list:
    """
    Analyzes contours in a binary image and computes properties such as bounding boxes,
    contour areas, and shape approximations (number of corners).

    Args:
        image (numpy.ndarray): A binary image (0 for background, 255 for foreground).
        area_threshold (int): Minimum contour area to consider. Contours with smaller area are ignored.

    Returns:
        list: A list of dictionaries containing contour properties (bounding box, area,
              corners, and approximation).
    """
    # Convert to grayscale if the image is not already single-channel
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []

    for cnt in contours:
        # Filter contours by area
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            continue  # Skip small contours

        # Bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Approximate the contour shape (number of corners)
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        results.append(
            {
                "bbox": (x, y, w, h),
                "area": area,
                "corners": len(approx),  # Approximation gives number of corners
                "approx": approx,
            }
        )

    return results


def draw_contours_and_bboxes(image: np.ndarray, contours_info: list) -> np.ndarray:
    """
    Draws contours, bounding boxes, and corners on the image.

    Args:
        image (numpy.ndarray): The image to draw on.
        contours_info (list): List of contour information to be drawn (output from analyze_contours).

    Returns:
        numpy.ndarray: The image with contours drawn.
    """
    for contour_info in contours_info:
        # Draw bounding box
        x, y, w, h = contour_info["bbox"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw contour approximation
        cv2.polylines(
            image,
            [contour_info["approx"]],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )

    return image


def process_and_display_image(image_path: str, area_threshold: int = AREA_THRESHOLD):
    """
    Processes an image, performs contour analysis, and displays the results.

    Args:
        image_path (str): Path to the image file to process.
        area_threshold (int): Minimum contour area to consider.
    """
    # Read the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get a binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Analyze contours in the binary image
    contours_info = analyze_contours(binary, area_threshold)

    # Draw contours and bounding boxes on the image
    result_image = draw_contours_and_bboxes(image.copy(), contours_info)

    # Display the image with contours and bounding boxes
    cv2.imshow("Contour Analysis", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Adjust image path as needed
    image_path = "./data/icub_ball_tracker.png"
    process_and_display_image(image_path)
