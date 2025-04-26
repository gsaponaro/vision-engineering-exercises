import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from package.pipeline_with_logging import (
    LoggedDataSource,
    LoggedProcessor,
    LoggedOutputStage,
    LoggedPipeline,
)


class TestLoggedPipeline(unittest.TestCase):

    # Decorators to replace the targets with mock objects during the test.
    # When the test is over, the original object is restored.
    @patch("package.pipeline_with_logging.LoggedDataSource")
    @patch("package.pipeline_with_logging.LoggedProcessor")
    @patch("package.pipeline_with_logging.LoggedOutputStage")
    def test_pipeline_runs_without_error(
        self, mock_output_stage_cls, mock_processor_cls, mock_data_source_cls
    ):
        """
        Test that the LoggedPipeline runs end-to-end without errors.

        We mock the components of the pipeline to isolate the test and
        ensure that the pipeline itself doesn't raise any exceptions when
        executing with these mock components.
        """
        # Mock the behavior of the data source, processor, and output stage
        mock_data_source = MagicMock()
        mock_processor = MagicMock()
        mock_output_stage = MagicMock()

        # Return mock objects for each component
        mock_data_source_cls.return_value = mock_data_source
        mock_processor_cls.return_value = mock_processor
        mock_output_stage_cls.return_value = mock_output_stage

        # Set up mock data to be processed
        mock_data_source.get_frame.side_effect = [
            np.zeros((2, 2, 3), dtype=np.uint8),  # First frame
            np.zeros((2, 2, 3), dtype=np.uint8),  # Second frame
            None,  # End of data
        ]
        mock_processor.process.return_value = np.ones((2, 2, 3), dtype=np.uint8)

        # Create the pipeline
        pipeline = LoggedPipeline(mock_data_source, mock_processor, mock_output_stage)

        # Run the pipeline and ensure no exceptions are raised
        try:
            pipeline.run()
        except Exception as e:
            self.fail(f"LoggedPipeline.run() raised an exception: {e}")

        # Assert that all components were called correctly
        mock_data_source.get_frame.assert_called()
        mock_processor.process.assert_called()
        mock_output_stage.output.assert_called()

    def test_logged_data_source_generates_correct_number_of_frames(self):
        """Test that LoggedDataSource generates the expected number of frames."""
        data_source = LoggedDataSource(num_frames=5)
        frames = []

        # Retrieve frames from the data source until None is returned
        while True:
            frame = data_source.get_frame()
            if frame is None:
                break
            frames.append(frame)

        # Assert the expected number of frames and their types
        self.assertEqual(len(frames), 5)
        self.assertTrue(all(isinstance(f, np.ndarray) for f in frames))

    def test_logged_processor_inverts_frame(self):
        """Test that LoggedProcessor correctly inverts a frame."""
        dummy_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        processor = LoggedProcessor(method="invert")
        processed_frame = processor.process(dummy_frame)

        # Inverting the frame should result in an array of 255s (max value for uint8)
        self.assertTrue(np.array_equal(processed_frame, 255 - dummy_frame))

    def test_logged_output_stage_does_not_modify_frame(self):
        """Test that LoggedOutputStage outputs frame metadata and does not alter the frame."""
        dummy_frame = np.ones((2, 2, 3), dtype=np.uint8) * 100
        output_stage = LoggedOutputStage()

        # Capture the original frame before output to check if it's altered
        original_frame = dummy_frame.copy()

        # Perform output (which should not alter the frame)
        output_stage.output(dummy_frame)

        # Assert that the original frame is not modified
        self.assertTrue(np.array_equal(dummy_frame, original_frame))


if __name__ == "__main__":
    unittest.main()
