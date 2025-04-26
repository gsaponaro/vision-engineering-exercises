"""
Exercise 6: Exception Handling and Logging

Take any of the previous exercises and add robust error handling:
- Input validation
- try/except blocks around critical code
- Use Python's logging module (not print) to log INFO, WARNING, and ERROR messages

Show an example log output with timestamp, level, and message

Topics: Logging and error handling
"""

import logging
import numpy as np
from typing import Optional
from .pipeline import DataSource, Processor, OutputStage, Pipeline

# Configure logging to show timestamp, level, and message
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LoggedDataSource(DataSource):
    """DataSource subclass that logs frame generation."""

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            frame = super().get_frame()
            if frame is not None:
                logging.debug(f"[DataSource] Generated frame {self.frames_generated}")
            else:
                logging.debug("[DataSource] No more frames to generate")
            return frame
        except Exception as e:
            logging.error(f"[DataSource] Error generating frame: {e}")
            raise


class LoggedProcessor(Processor):
    """Processor subclass that logs processing operations."""

    def process(self, frame: np.ndarray) -> np.ndarray:
        try:
            logging.debug(f"[Processor] Processing frame with method: {self.method}")
            return super().process(frame)
        except Exception as e:
            logging.error(f"[Processor] Error processing frame: {e}")
            raise


class LoggedOutputStage(OutputStage):
    """OutputStage subclass that logs output metadata."""

    def output(self, processed_frame: np.ndarray) -> None:
        try:
            metadata = {"shape": processed_frame.shape, "dtype": processed_frame.dtype}
            logging.info(f"[OutputStage] Output metadata: {metadata}")
            return super().output(processed_frame)
        except Exception as e:
            logging.error(f"[OutputStage] Error outputting frame: {e}")
            raise


class LoggedPipeline(Pipeline):
    """Pipeline subclass that logs overall pipeline execution."""

    def run(self):
        try:
            logging.info("[Pipeline] Starting pipeline")
            super().run()
            logging.info("[Pipeline] Finished pipeline")
        except Exception as e:
            logging.error(f"[Pipeline] Error during pipeline execution: {e}")
            raise
