"""
detector.py
-----------
Deterministic vision detector for localizing a white sugar cube in camera frames.
Compatible with both simulated (MuJoCo) and physical (CSI/USB) camera streams.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2


@dataclass
class TargetDetection:
    """
    Structured visual detection result.
    """
    detected: bool = False

    # Normalized image coordinates in [-1.0, 1.0]
    # center_x: -1.0 = left, 0.0 = center, +1.0 = right
    # center_y: -1.0 = above (top of image), 0.0 = center, +1.0 = below (bottom of image)
    center_x: float = 0.0
    center_y: float = 0.0

    # Normalized dimensions in [0.0, 1.0]
    width: float = 0.0
    height: float = 0.0
    apparent_size: float = 0.0  # normalized bounding area

    confidence: float = 0.0

    # Categorical descriptors for biological sensory representation
    horizontal_position: str = "center"  # "left", "center", "right"
    vertical_position: str = "center"    # "above", "center", "below"
    size_category: str = "small"         # "small", "medium", "large"

    # Raw pixel coordinates
    pixel_x: int = 0
    pixel_y: int = 0
    bbox_pixels: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)


class WhiteSugarDetector:
    """
    Detects white target objects (sugar cube) in RGB camera frames using
    color thresholding and contour localization.
    """

    def __init__(
        self,
        min_brightness: int = 180,
        max_saturation: int = 50,
        min_area_pixels: int = 15,
        deadband_norm: float = 0.18,
    ):
        """
        Parameters
        ----------
        min_brightness : int
            Minimum intensity in V/RGB channel to qualify as white (0-255).
        max_saturation : int
            Maximum color saturation in HSV to qualify as neutral white (0-255).
        min_area_pixels : int
            Minimum pixel area threshold to reject noise.
        deadband_norm : float
            Normalized image threshold separating 'center' from 'left'/'right'.
        """
        self.min_brightness = min_brightness
        self.max_saturation = max_saturation
        self.min_area_pixels = min_area_pixels
        self.deadband_norm = deadband_norm

    def detect(self, frame_rgb: np.ndarray) -> TargetDetection:
        """
        Process an RGB image frame and detect the white sugar cube.

        Parameters
        ----------
        frame_rgb : np.ndarray
            RGB image of shape (H, W, 3) and uint8 dtype.

        Returns
        -------
        TargetDetection
            Detection parameters and categorical features.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return TargetDetection(detected=False)

        h_img, w_img = frame_rgb.shape[:2]
        total_img_area = float(h_img * w_img)

        # Convert RGB -> HSV for robust white segmentation
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = cv2.split(frame_hsv)

        # White mask: high brightness, low saturation
        white_mask = (v_ch >= self.min_brightness) & (s_ch <= self.max_saturation)
        mask_uint8 = (white_mask.astype(np.uint8)) * 255

        # Morphological opening to clean single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

        # Find external contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return TargetDetection(detected=False)

        # Select the most prominent contour by area
        best_contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(best_contour))

        if area < self.min_area_pixels:
            return TargetDetection(detected=False)

        # Bounding box and centroid
        x, y, w, h = cv2.boundingRect(best_contour)
        moments = cv2.moments(best_contour)
        if moments["m00"] > 0:
            cx_px = int(moments["m10"] / moments["m00"])
            cy_px = int(moments["m01"] / moments["m00"])
        else:
            cx_px = x + w // 2
            cy_px = y + h // 2

        # Compute normalized coordinates in [-1.0, 1.0]
        # center_x: 0.0 at image center, -1.0 at left, +1.0 at right
        norm_cx = (cx_px - (w_img / 2.0)) / (w_img / 2.0)
        # center_y: 0.0 at image center, -1.0 at top (above), +1.0 at bottom (below)
        norm_cy = (cy_px - (h_img / 2.0)) / (h_img / 2.0)

        norm_w = float(w) / w_img
        norm_h = float(h) / h_img
        apparent_size = float(w * h) / total_img_area

        # Confidence based on size and fill ratio
        fill_ratio = area / max(1.0, float(w * h))
        confidence = float(np.clip(fill_ratio * min(1.0, area / 100.0), 0.1, 1.0))

        # Categorical position descriptors
        if norm_cx < -self.deadband_norm:
            horiz_pos = "left"
        elif norm_cx > self.deadband_norm:
            horiz_pos = "right"
        else:
            horiz_pos = "center"

        if norm_cy < -self.deadband_norm:
            vert_pos = "above"
        elif norm_cy > self.deadband_norm:
            vert_pos = "below"
        else:
            vert_pos = "center"

        # Categorical size descriptors
        if apparent_size < 0.005:
            size_cat = "small"
        elif apparent_size > 0.035:
            size_cat = "large"
        else:
            size_cat = "medium"

        return TargetDetection(
            detected=True,
            center_x=float(np.clip(norm_cx, -1.0, 1.0)),
            center_y=float(np.clip(norm_cy, -1.0, 1.0)),
            width=norm_w,
            height=norm_h,
            apparent_size=apparent_size,
            confidence=confidence,
            horizontal_position=horiz_pos,
            vertical_position=vert_pos,
            size_category=size_cat,
            pixel_x=cx_px,
            pixel_y=cy_px,
            bbox_pixels=(x, y, w, h),
        )
