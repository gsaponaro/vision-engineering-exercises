import unittest
import numpy as np
from package.pipeline import DataSource, Processor, OutputStage, Pipeline


class TestDataSource(unittest.TestCase):
    def test_frame_generation(self):
        """Test that DataSource stops after num_frames and produces the correct shape."""
        ds = DataSource(num_frames=3, frame_size=(50, 50))
        frames = [ds.get_frame() for _ in range(5)]  # Try to get more than available
        valid_frames = [f for f in frames if f is not None]

        self.assertEqual(len(valid_frames), 3)
        self.assertTrue(all(f.shape == (50, 50, 3) for f in valid_frames))

        # The fact that we only got 3 valid frames confirms frames_generated capped output


class TestProcessor(unittest.TestCase):
    def test_edge_detection_output(self):
        """Test the Canny edge detector produces expected output type and shape."""
        dummy_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        proc = Processor(method="edges")
        result = proc.process(dummy_frame)
        self.assertEqual(result.shape, (100, 100))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all((result == 0) | (result == 255)))  # Binary output


class TestOutputStage(unittest.TestCase):
    def test_logging(self):
        """Ensure metadata logging works and tracks output correctly."""
        output = OutputStage(display=False)
        dummy = np.zeros((60, 60), dtype=np.uint8)
        output.output(dummy)
        self.assertEqual(len(output.logged), 1)
        self.assertEqual(output.logged[0]["shape"], (60, 60))
        self.assertEqual(output.logged[0]["dtype"], np.uint8)


class TestPipeline(unittest.TestCase):
    def test_pipeline_integration(self):
        """Integration test: run full pipeline and validate output metadata."""
        ds = DataSource(num_frames=2, frame_size=(30, 30))
        proc = Processor()
        out = OutputStage()
        pipe = Pipeline(ds, proc, out)
        pipe.run()
        self.assertEqual(len(out.logged), 2)
        for entry in out.logged:
            self.assertEqual(entry["shape"], (30, 30))


if __name__ == "__main__":
    unittest.main()
