import unittest
from types import SimpleNamespace

import numpy as np

from local_app import vanish_cam as app
from local_app.goku_effect import goku_pose_from_landmarks, render_teleport


def face():
    points = [SimpleNamespace(x=0.5, y=0.4) for _ in range(468)]
    points[234] = SimpleNamespace(x=0.4, y=0.4)
    points[454] = SimpleNamespace(x=0.6, y=0.4)
    points[10] = SimpleNamespace(x=0.5, y=0.25)
    points[151] = SimpleNamespace(x=0.5, y=0.28)
    points[9] = SimpleNamespace(x=0.5, y=0.33)
    return points


def hand(index_x=0.497, middle_x=0.505, dy=0.0, ring_y=0.47):
    points = [SimpleNamespace(x=0.5, y=0.55 + dy) for _ in range(21)]
    for index, x in ((5, 0.49), (9, 0.5), (13, 0.51), (17, 0.52)):
        points[index] = SimpleNamespace(x=x, y=0.45 + dy)
    points[8] = SimpleNamespace(x=index_x, y=0.29 + dy)
    points[12] = SimpleNamespace(x=middle_x, y=0.29 + dy)
    points[16] = SimpleNamespace(x=0.51, y=ring_y + dy)
    points[20] = SimpleNamespace(x=0.52, y=ring_y + dy)
    return points


class GokuPoseTests(unittest.TestCase):
    def test_two_fingers_on_forehead(self):
        self.assertTrue(goku_pose_from_landmarks([hand()], face(), 640, 480))

    def test_other_poses_are_rejected(self):
        self.assertFalse(goku_pose_from_landmarks([hand()], None, 640, 480))
        self.assertFalse(goku_pose_from_landmarks([], face(), 640, 480))
        self.assertFalse(goku_pose_from_landmarks([hand(0.44, 0.56)], face(), 640, 480))  # V
        self.assertFalse(goku_pose_from_landmarks([hand(dy=0.3)], face(), 640, 480))  # Sul petto
        self.assertFalse(goku_pose_from_landmarks([hand(ring_y=0.3)], face(), 640, 480))  # Mano aperta


class GokuHoldTests(unittest.TestCase):
    def test_pose_must_be_held_and_released_between_teleports(self):
        state = app.AppState(phase="ready")
        hold = app.GOKU_HOLD_SECONDS
        self.assertFalse(app.update_goku_hold(state, True, 1.0))
        self.assertFalse(app.update_goku_hold(state, False, 1.5))  # Interrotto.
        self.assertFalse(app.update_goku_hold(state, True, 2.0))
        self.assertTrue(app.update_goku_hold(state, True, 2.0 + hold))
        # Restando in posa non si riappare subito.
        self.assertFalse(app.update_goku_hold(state, True, 4.0 + hold))
        self.assertFalse(app.update_goku_hold(state, False, 5.0))
        self.assertFalse(app.update_goku_hold(state, True, 5.1))
        self.assertTrue(app.update_goku_hold(state, True, 5.1 + hold))


class TeleportRenderTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((60, 80, 3), 200, np.uint8)
        self.background = np.full((60, 80, 3), 10, np.uint8)
        self.mask = np.zeros((60, 80), np.float32)
        self.mask[10:50, 30:50] = 1.0

    def test_vanish_ends_on_background_and_appear_ends_on_person(self):
        start = render_teleport(self.frame, self.background, self.mask, 0.0)
        np.testing.assert_array_equal(start[20, 40], self.frame[20, 40])
        end = render_teleport(self.frame, self.background, self.mask, 1.0)
        np.testing.assert_array_equal(end, self.background)
        back = render_teleport(self.frame, self.background, self.mask, 1.0, appearing=True)
        np.testing.assert_array_equal(back[20, 40], self.frame[20, 40])

    def test_speed_lines_are_drawn_mid_effect(self):
        middle = render_teleport(self.frame, self.background, self.mask, 0.5)
        self.assertGreater(int(np.abs(middle.astype(int) - self.background).sum()), 0)
        outside = middle[:, :5]
        self.assertTrue(np.array_equal(outside, self.background[:, :5]))


if __name__ == "__main__":
    unittest.main()
