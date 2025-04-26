"""
Exercise 5: Real-Time Pipeline Skeleton

Write a modular pipeline with the following components:
- Data source (e.g., simulated sensor stream or video frames)
- Processing stage (e.g., basic smoothing or edge detection)
- Output stage (e.g., print/log result or save image)

Use Object-Oriented Programming (OOP) to organize your code and demonstrate how
you'd isolate testable units.

Topics: Architecture design
"""

from typing import Optional
import cv2
import numpy as np


class DataSource:
    """Simulates a real-time data source like a camera or sensor stream."""

    def __init__(self, num_frames: int = 10, frame_size: tuple[int, int] = (100, 100)):
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.frames_generated = 0  # Private property, not exposed

    def get_frame(self) -> Optional[np.ndarray]:
        """Returns a synthetic frame (a solid color image). None when done."""
        if self.frames_generated >= self.num_frames:
            return None
        self.frames_generated += 1
        # Generates a synthetic RGB image with a gradient
        frame = np.full(
            (self.frame_size[0], self.frame_size[1], 3), 128, dtype=np.uint8
        )
        cv2.circle(
            frame,
            (self.frames_generated * 5 % self.frame_size[1], self.frame_size[0] // 2),
            10,
            (255, 255, 255),
            -1,
        )
        return frame


class Processor:
    """Applies a transformation to the input frame, e.g., smoothing or edge detection."""

    def __init__(self, method: str = "edges"):
        self.method = method

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Processes the input frame and returns the result."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.method == "edges":
            # Canny edge detector
            return cv2.Canny(gray, 50, 150)
        elif self.method == "blur":
            return cv2.GaussianBlur(gray, (5, 5), 0)
        elif self.method == "invert":
            return 255 - frame
        else:
            raise ValueError(f"Unknown processing method: {self.method}")


class OutputStage:
    """Handles outputting the result, e.g., displaying, saving, or logging it."""

    def __init__(self, display: bool = False):
        self.display = display
        self.logged = []  # For testing

    def output(self, processed_frame: np.ndarray) -> None:
        """Logs metadata and optionally shows the result."""
        self.logged.append(
            {"shape": processed_frame.shape, "dtype": processed_frame.dtype}
        )
        if self.display:
            cv2.imshow("Output", processed_frame)
            cv2.waitKey(1)


class Pipeline:
    """
    Combines DataSource, Processor, and OutputStage into a cohesive pipeline.
    Demonstrates modular design and separation of concerns.
    """

    def __init__(self, source: DataSource, processor: Processor, output: OutputStage):
        self.source = source
        self.processor = processor
        self.output = output

    def run(self):
        """Main loop: fetch, process, and output frames until exhausted."""
        while True:
            frame = self.source.get_frame()
            if frame is None:
                break
            processed = self.processor.process(frame)
            self.output.output(processed)
