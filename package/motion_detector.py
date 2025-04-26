"""
Exercise 3: Motion Detection in Video

Using OpenCV, write a script that:
- Captures webcam frames
- Applies frame differencing for motion detection
- Displays bounding boxes around moving regions
- Skips processing if no significant motion is detected

Include proper logging and clean code structure.

Topics: Video processing basics
"""

import cv2
import logging
import os

# Suppress the error message related to missing platform plugins in Qt
# qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Motion sensitivity constants
THRESHOLD: int = 25  # Pixel intensity change required to consider motion
MIN_AREA: int = 500  # Minimum area in pixels for a contour to be considered motion

# Setup logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def setup_camera() -> cv2.VideoCapture:
    """
    Sets up the webcam for motion detection.

    Returns:
        cv2.VideoCapture: The video capture object for the webcam.

    Raises:
        RuntimeError: If the webcam cannot be opened.
    """
    camera_index: int = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Cannot open webcam")
        raise RuntimeError("Webcam initialization failed")
    return cap


def detect_motion(
    prev_frame, current_frame, threshold=THRESHOLD, min_area=MIN_AREA
) -> list:
    """
    Detects motion by comparing the current and previous frames.

    Args:
        prev_frame (numpy.ndarray): The previous frame in grayscale.
        current_frame (numpy.ndarray): The current frame in grayscale.
        threshold (int): The pixel intensity change required to consider motion.
        min_area (int): The minimum area in pixels for a contour to be considered motion.

    Returns:
        list: A list of bounding boxes for the regions with detected motion.
    """
    # Compute the absolute difference between the previous and current grayscale frames.
    frame_delta = cv2.absdiff(prev_frame, current_frame)

    # Apply binary thresholding to the delta image to emphasize motion areas.
    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Extract contours (connected components) and keep only those above the minimum area.
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    motion_regions = [
        cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area
    ]

    return motion_regions


def draw_bounding_boxes(frame, regions):
    """
    Draws bounding boxes around detected motion regions.

    Args:
        frame (numpy.ndarray): The current frame of the video.
        regions (list): A list of bounding boxes to draw.
    """
    for x, y, w, h in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def run_motion_detector():
    """
    Runs the motion detection system, processing webcam frames in real-time.
    Detects motion and displays bounding boxes around the moving regions.
    """
    cap = setup_camera()
    prev_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from webcam")
                break

            # Preprocess frame: convert to grayscale.
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is None:
                prev_frame = processed
                continue

            # Detect motion between the previous and current frame
            motion_regions = detect_motion(prev_frame, processed)
            prev_frame = processed

            if motion_regions:
                logging.info(f"Motion detected in {len(motion_regions)} region(s)")
                draw_bounding_boxes(frame, motion_regions)
            else:
                logging.debug("No significant motion detected")

            cv2.imshow("Motion Detection", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Exiting...")
                break

    finally:
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_motion_detector()
