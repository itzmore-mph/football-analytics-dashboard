from __future__ import annotations

import math

from src.features_xg import shot_angle, shot_distance


def test_shot_distance_center():
    assert math.isclose(shot_distance(120, 40), 0.0, abs_tol=1e-6)


def test_shot_angle_limits():
    assert 0.0 <= shot_angle(100, 40) <= math.pi
