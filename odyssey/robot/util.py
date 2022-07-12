"""
util.py

Utility classes and functions for facilitating robot control via Polymetis' built-in RobotInterface.
"""
import time

import numpy as np


# Constants
HZ, POLE_LIMIT, TOLERANCE = 20, (1.0 - 1e-6), 1e-10

# fmt: off
HOMES = {
    "default": [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],
}

# Joint Controller gains -- we want a compliant robot when recording, and stiff when playing back / teleoperating
KQ_GAINS = {
    "record": [1, 1, 1, 1, 1, 1, 1],
    "rb2": [26.6667, 40.0000, 33.3333, 33.3333, 23.3333, 16.6667, 6.6667],
    "default": [80, 120, 100, 100, 70, 50, 20],
    "teleoperate": [240.0, 360.0, 300.0, 300.0, 210.0, 150.0, 60.0],
}
KQD_GAINS = {
    "record": [1, 1, 1, 1, 1, 1, 1],
    "rb2": [3.3333, 3.3333, 3.3333, 3.3333, 1.6667, 1.6667, 1.6667],
    "default": [10, 10, 10, 10, 5, 5, 5],
    "teleoperate": [30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0],
}

# End-Effector Controller gains -- we want a compliant robot when recording, and stiff when playing back / operating
KX_GAINS = {
    "record": [1, 1, 1, 1, 1, 1],
    "default": [150, 150, 150, 10, 10, 10],
    "teleoperate": [450, 450, 450, 30, 30, 30],     # Upper bound is ???, so using joint default -- 3x multiplier
}
KXD_GAINS = {
    "record": [1, 1, 1, 1, 1, 1],
    "default": [25, 25, 25, 7, 7, 7],
    "teleoperate": [75, 75, 75, 21, 21, 21],        # Upper bound is ???, so using joint default -- 3x multiplier
}
# fmt: off


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the robot control loop.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()


# Rotation & Quaternion Helpers
#   =>> Reference: DM Control -- https://github.com/deepmind/dm_control/blob/main/dm_control/utils/transformations.py
def quat2rmat(quat) -> np.ndarray:
    """
    Return homogeneous rotation matrix from quaternion.

    Args:
      quat: A quaternion [w, i, j, k].

    Returns:
      A 4x4 homogeneous matrix with the rotation corresponding to `quat`.
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < TOLERANCE:
        return np.identity(4)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)

    return np.array(
        (
            (1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0),
            (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0),
            (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ),
        dtype=np.float64
    )


def rmat2euler_xyz(rmat: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to XYZ euler angles.

    | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
    | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
    | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
    """
    if rmat[0, 2] > POLE_LIMIT:
        print("[Warning =>> quat2euler] :: Angle at North Pole")
        z = np.arctan2(rmat[1, 0], rmat[1, 1])
        y = np.pi / 2
        x = 0.0
        return np.array([x, y, z])

    elif rmat[0, 2] < -POLE_LIMIT:
        print("[Warning =>> quat2euler] :: Angle at South Pole")
        z = np.arctan2(rmat[1, 0], rmat[1, 1])
        y = -np.pi / 2
        x = 0.0
        return np.array([x, y, z])

    else:
        z = -np.arctan2(rmat[0, 1], rmat[0, 0])
        y = np.arcsin(rmat[0, 2])
        x = -np.arctan2(rmat[1, 2], rmat[2, 2])
        return np.array([x, y, z])


def quat2euler(quat: np.ndarray, ordering="XYZ"):
    """
    Returns the Euler angles corresponding to the provided quaternion.

    Args:
      quat: A quaternion [w, i, j, k].
      ordering: (str) Desired euler angle ordering.

    Returns:
      euler_vec: The euler angle rotations.

    Reference: DM Control -- https://github.com/deepmind/dm_control/blob/main/dm_control/utils/transformations.py
    """
    mat = quat2rmat(quat)
    if ordering == "XYZ":
        return rmat2euler_xyz(mat[0:3, 0:3])
    else:
        raise NotImplementedError("Quat2Euler for Ordering != XYZ is not yet defined!")
