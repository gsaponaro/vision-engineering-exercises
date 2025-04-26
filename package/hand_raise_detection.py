"""
Bonus Exercise:
Implement a system that detects if a person raises their hand above shoulder level based on keypoints.
Use simulated or real-time video (OpenCV + MediaPipe or pre-recorded video).
"""

import cv2
import mediapipe as mp
import os
import logging

# Suppress platform plugin warnings
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Set up logging configuration
logging.basicConfig(level=logging.INFO)


class HandRaiseDetector:
    def __init__(self):
        # Initialize MediaPipe Pose detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        logging.info("HandRaiseDetector initialized with MediaPipe Pose.")

    def is_hand_raised(self, landmarks):
        """
        Check if either hand is raised above shoulder level using the pose landmarks.

        Parameters:
            landmarks: List of pose landmarks from MediaPipe.

        Returns:
            bool: True if hand is raised, False otherwise.
        """
        # Get coordinates of keypoints
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Check if the wrist is above the shoulder on the y-axis
        left_hand_raised = left_wrist.y < left_shoulder.y
        right_hand_raised = right_wrist.y < right_shoulder.y

        # Log the state of hand raise detection
        logging.debug(
            f"Left hand raised: {left_hand_raised}, Right hand raised: {right_hand_raised}"
        )

        return left_hand_raised or right_hand_raised

    def process_frame(self, frame):
        """
        Process a video frame and detect hand raise.

        Parameters:
            frame: Input video frame.

        Returns:
            tuple: The processed frame and a boolean indicating if a hand is raised.
        """
        # Flip the frame horizontally for a later selfie view
        frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV default) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw pose landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

            # Detect if a hand is raised
            hand_raised = self.is_hand_raised(results.pose_landmarks.landmark)

            # Log detection result (debug because it's very verbose)
            logging.debug(f"Hand raised: {hand_raised}")

            return frame, hand_raised

        return frame, False


def main():
    detector = HandRaiseDetector()

    # Start video input using webcam (or replace 0 with video file path for file input)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logging.error("Failed to open video capture device.")
        return

    logging.info("Video capture started.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from video capture.")
            break

        # Process the frame for hand raise detection
        frame, hand_raised = detector.process_frame(frame)

        # Display the appropriate text and color based on hand raise status
        text = "Hand Raised!" if hand_raised else "Hand Not Raised"
        color = (0, 255, 0) if hand_raised else (0, 0, 255)
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the frame
        cv2.imshow("Hand Raise Detection", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logging.info("Exiting on 'q' key press.")
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Resources released, program finished.")


if __name__ == "__main__":
    main()
