"""
Exercise 2: Pose Angle Calculator from Keypoints

Given a list of 2D keypoints representing human joints (e.g., shoulder, elbow,
wrist), implement a function to compute the elbow flexion angle (i.e., angle
between shoulder-elbow-wrist, in degrees).

Example:

keypoints = { "shoulder": (x1, y1), "elbow": (x2, y2), "wrist": (x3, y3) }
calculate_angle(keypoints)

Topics: Geometry on body keypoints
"""

import math
from typing import Dict, Tuple


def calculate_angle(keypoints: Dict[str, Tuple[float, float]]) -> float:
    """
    Calculates the elbow flexion angle in degrees from 2D keypoints.

    The angle is defined between the shoulder, elbow, and wrist keypoints.

    Args:
        keypoints: A dictionary with keys "shoulder", "elbow", and "wrist",
                   each mapping to an (x, y) coordinate tuple.

    Returns:
        The angle in degrees between shoulder-elbow-wrist.

    Raises:
        KeyError: If any of the required keypoints are missing.
    """

    def angle_between(
        p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
    ) -> float:
        a = (p1[0] - p2[0], p1[1] - p2[1])
        b = (p3[0] - p2[0], p3[1] - p2[1])
        dot_product = a[0] * b[0] + a[1] * b[1]
        # Unpack tuple (vector), then compute its magnitude (Euclidean norm)
        mag_a = math.hypot(*a)
        mag_b = math.hypot(*b)
        if mag_a * mag_b == 0:
            return 0.0
        cosine = dot_product / (mag_a * mag_b)
        cosine = max(min(cosine, 1.0), -1.0)  # Clamp for numerical stability
        return math.degrees(math.acos(cosine))

    return angle_between(keypoints["shoulder"], keypoints["elbow"], keypoints["wrist"])
