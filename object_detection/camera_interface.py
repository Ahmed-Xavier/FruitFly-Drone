"""
camera_interface.py
-------------------
Unified camera interface supporting both MuJoCo simulation rendering
and physical USB/CSI camera hardware for real-world drone experiments.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class CameraInterface(ABC):
    """Abstract base class for vision capture."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """
        Capture and return a single frame.

        Returns
        -------
        np.ndarray
            Image frame of shape (height, width, 3) with uint8 RGB values.
        """
        pass

    @property
    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        pass

    def close(self) -> None:
        """Release camera resources if applicable."""
        pass


class MuJoCoCamera(CameraInterface):
    """
    Simulated camera rendering directly from a named MuJoCo camera site.
    """

    def __init__(
        self,
        model,
        data,
        camera_name: str = "drone_pov",
        width: int = 320,
        height: int = 240,
    ):
        import mujoco
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height

        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.cam_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")

        self.renderer = mujoco.Renderer(model, height=height, width=width)

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def get_frame(self) -> np.ndarray:
        """Render and return current POV camera image in RGB format."""
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return self.renderer.render()

    def close(self) -> None:
        if hasattr(self.renderer, "close"):
            self.renderer.close()


class PhysicalCamera(CameraInterface):
    """
    Physical hardware camera interface using OpenCV VideoCapture (USB/CSI camera).
    """

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        import cv2
        self.device_index = device_index
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            print(f"[PhysicalCamera] Warning: Camera device {device_index} could not be opened.")

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def get_frame(self) -> np.ndarray:
        """Capture frame from hardware camera and convert to RGB format."""
        import cv2
        if not self.cap.isOpened():
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self.cap.isOpened():
            self.cap.release()


class MockCamera(CameraInterface):
    """
    Synthetic mock camera for headless and unit test environments.
    """

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        target_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Parameters
        ----------
        target_box : tuple, optional
            (x, y, w, h) bounding box of a synthetic white target, or None for empty scene.
        """
        self.width = width
        self.height = height
        self.target_box = target_box

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def set_target_box(self, target_box: Optional[Tuple[int, int, int, int]]) -> None:
        self.target_box = target_box

    def get_frame(self) -> np.ndarray:
        """Generate a synthetic dark background with optional white target box."""
        frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)  # dark background
        if self.target_box is not None:
            x, y, w, h = self.target_box
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(self.width, x + w)
            y2 = min(self.height, y + h)
            frame[y1:y2, x1:x2] = 255  # pure white target
        return frame
