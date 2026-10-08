"""
test_sugar_vision.py
--------------------
Lightweight test suite for the visual sugar-cube tracking and sensory pipeline.
Validates:
  1. Sugar cube physical and visual presence in MuJoCo model
  2. Camera interface rendering (MuJoCoCamera & MockCamera)
  3. Deterministic white sugar detection
  4. Accurate image-space localization (left, center, right, apparent size)
  5. Correct handling of 'not detected' / absent target
  6. VisualTargetEncoder translation into LC10 neuron firing rates
  7. Regression preservation of the pure gustatory brain 'sugar' stimulus
  8. Closed-loop integrated stepping with visual target input

Can be run quickly on CPU without requiring a GPU:
    python tests/test_sugar_vision.py
    pytest tests/test_sugar_vision.py
"""

import sys
import unittest
from pathlib import Path
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from body import Body
from brain import Brain, FlyBrain, STIMULI
from bridge import BrainDroneSynchronizer, VisualTargetEncoder
from object_detection import (
    TargetConfig,
    WhiteSugarDetector,
    TargetDetection,
    MuJoCoCamera,
    MockCamera,
    get_target_position,
    set_target_position,
    get_relative_target_vector,
)


class TestSugarVisionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body = Body()

    def test_01_sugar_cube_exists_in_mujoco(self):
        """Verify the sugar cube body, geom, and site exist in the MuJoCo model."""
        import mujoco
        body_id = mujoco.mj_name2id(self.body.model, mujoco.mjtObj.mjOBJ_BODY, "sugar_cube")
        self.assertGreaterEqual(body_id, 0, "Body 'sugar_cube' must exist in MuJoCo model.")

        geom_id = mujoco.mj_name2id(self.body.model, mujoco.mjtObj.mjOBJ_GEOM, "sugar_geom")
        self.assertGreaterEqual(geom_id, 0, "Geom 'sugar_geom' must exist in MuJoCo model.")

        site_id = mujoco.mj_name2id(self.body.model, mujoco.mjtObj.mjOBJ_SITE, "sugar_site")
        self.assertGreaterEqual(site_id, 0, "Site 'sugar_site' must exist in MuJoCo model.")

        # Test querying position
        pos = get_target_position(self.body.model, self.body.data, "sugar_cube")
        self.assertEqual(len(pos), 3)

        # Test repositioning
        new_pos = [3.0, 0.5, 1.2]
        set_target_position(self.body.model, self.body.data, new_pos, "sugar_cube")
        updated_pos = get_target_position(self.body.model, self.body.data, "sugar_cube")
        np.testing.assert_allclose(updated_pos, new_pos, atol=1e-3)

    def test_02_camera_returns_image(self):
        """Verify camera interface captures valid RGB image arrays."""
        # Test MockCamera
        mock_cam = MockCamera(width=160, height=120)
        frame_mock = mock_cam.get_frame()
        self.assertEqual(frame_mock.shape, (120, 160, 3))
        self.assertEqual(frame_mock.dtype, np.uint8)

        # Test MuJoCoCamera
        cam = MuJoCoCamera(self.body.model, self.body.data, camera_name="drone_pov", width=160, height=120)
        frame = cam.get_frame()
        self.assertEqual(frame.shape, (120, 160, 3))
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(frame.mean(), 0, "Rendered camera frame should not be completely black.")

    def test_03_detector_detects_white_sugar_cube(self):
        """Verify WhiteSugarDetector accurately detects a white target."""
        detector = WhiteSugarDetector()
        mock_cam = MockCamera(width=320, height=240, target_box=(140, 100, 40, 40))
        frame = mock_cam.get_frame()
        det = detector.detect(frame)

        self.assertTrue(det.detected)
        self.assertGreater(det.confidence, 0.5)
        self.assertAlmostEqual(det.center_x, 0.0, delta=0.15)
        self.assertAlmostEqual(det.center_y, 0.0, delta=0.15)
        self.assertEqual(det.horizontal_position, "center")

    def test_04_detector_left_right_spatial_localization(self):
        """Verify detector discriminates left, right, and center positions."""
        detector = WhiteSugarDetector()

        # Target on Left
        mock_left = MockCamera(width=320, height=240, target_box=(30, 100, 30, 30))
        det_left = detector.detect(mock_left.get_frame())
        self.assertTrue(det_left.detected)
        self.assertLess(det_left.center_x, -0.2)
        self.assertEqual(det_left.horizontal_position, "left")

        # Target on Right
        mock_right = MockCamera(width=320, height=240, target_box=(260, 100, 30, 30))
        det_right = detector.detect(mock_right.get_frame())
        self.assertTrue(det_right.detected)
        self.assertGreater(det_right.center_x, 0.2)
        self.assertEqual(det_right.horizontal_position, "right")

    def test_05_detector_handles_absent_target(self):
        """Verify detector returns detected=False when target is absent."""
        detector = WhiteSugarDetector()
        mock_empty = MockCamera(width=320, height=240, target_box=None)
        det_empty = detector.detect(mock_empty.get_frame())
        self.assertFalse(det_empty.detected)
        self.assertEqual(det_empty.confidence, 0.0)

    def test_06_target_encoder_lc10_pathway(self):
        """Verify VisualTargetEncoder maps detections to bilateral LC10 firing rates."""
        encoder = VisualTargetEncoder()
        self.assertGreater(len(encoder.lc10_left_ids), 0)
        self.assertGreater(len(encoder.lc10_right_ids), 0)

        # 1. Quiescent baseline when not detected
        no_det = TargetDetection(detected=False)
        enc_no = encoder.encode(no_det)
        self.assertFalse(enc_no["summary"]["target_detected"])
        self.assertEqual(enc_no["summary"]["lc10_left_rate_hz"], encoder.BASELINE_RATE_HZ)
        self.assertEqual(enc_no["summary"]["lc10_right_rate_hz"], encoder.BASELINE_RATE_HZ)

        # 2. Left target excitation asymmetry
        left_det = TargetDetection(detected=True, center_x=-0.6, apparent_size=0.03, confidence=0.9)
        enc_left = encoder.encode(left_det)
        self.assertTrue(enc_left["summary"]["target_detected"])
        self.assertGreater(
            enc_left["summary"]["lc10_left_rate_hz"],
            enc_left["summary"]["lc10_right_rate_hz"],
            "Left target must stimulate Left LC10 more strongly than Right LC10",
        )

        # 3. Right target excitation asymmetry
        right_det = TargetDetection(detected=True, center_x=+0.6, apparent_size=0.03, confidence=0.9)
        enc_right = encoder.encode(right_det)
        self.assertGreater(
            enc_right["summary"]["lc10_right_rate_hz"],
            enc_right["summary"]["lc10_left_rate_hz"],
            "Right target must stimulate Right LC10 more strongly than Left LC10",
        )

    def test_07_existing_brain_sugar_stimulus_preserved(self):
        """Regression test: verify python main.py --brain --stimulus sugar remains functional."""
        self.assertIn("sugar", STIMULI)
        sugar_info = STIMULI["sugar"]
        self.assertEqual(len(sugar_info["neurons"]), 21)
        self.assertIn("Sugar GRNs", sugar_info["description"])

        # Test stepping brain with 'sugar' stimulus on CPU
        brain = FlyBrain(device="cpu", plasticity=False)
        out = brain.step({"stimulus": "sugar"})
        self.assertIn("spikes", out)
        self.assertIn("dn_rates", out)
        self.assertIn("population_rates", out)

    def test_08_integrated_loop_with_target_stepping(self):
        """Verify BrainDroneSynchronizer executes synchronized steps with visual target input."""
        # Use CPU brain for fast headless unit test
        brain = FlyBrain(device="cpu", plasticity=False)
        sync = BrainDroneSynchronizer(
            body=self.body,
            brain=brain,
            neural_steps_per_body_step=1,
            target_alt=1.0,
        )

        detection = TargetDetection(detected=True, center_x=0.0, apparent_size=0.02, confidence=0.8)
        snap = sync.step(target_detection=detection)

        self.assertIsNotNone(snap.target_summary)
        self.assertTrue(snap.target_summary["target_detected"])
        self.assertFalse(snap.safety_triggered)
        self.assertEqual(len(snap.motor_commands), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
