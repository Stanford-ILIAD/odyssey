"""
rotations.py

Standalone verification script to evaluate the differences between the default Scipy Rotation implementation
(quat/euler), the TorchControl (Polymetis) Rotation implementation (derived from Scipy), and the separate DM Control
utility functions.
"""
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


# CONSTANTS -- EE Orientation is ~ what gets returned when you run FK on the IRIS "home" pose...
HZ, POLE_LIMIT, TOLERANCE = 20, (1.0 - 1e-6), 1e-10
EE_ORIENTATION = np.array([0.02872275, 0.00059619, -0.01118877])


# DM Control :: Rotation & Quaternion Helpers
#   =>> Reference: DM Control -- https://github.com/deepmind/dm_control/blob/main/dm_control/utils/transformations.py
def quat2rmat(quat: np.ndarray) -> np.ndarray:
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
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def axis_rotation(theta):
    """Returns the theta dim, cos and sin, and blank matrix for axis rotation."""
    n = 1 if np.isscalar(theta) else len(theta)
    ct, st = np.cos(theta), np.sin(theta)
    rmat = np.zeros((n, 3, 3))
    return n, ct, st, rmat


def rotation_x_axis(theta):
    """
    Returns a rotation matrix of a rotation about the X-axis.

    Supports vector-valued theta, in which case the returned array is of shape (len(t), 3, 3), or (len(t), 4, 4) if
    full=True. If theta is scalar the batch dimension is squeezed out.

    Args:
      theta: The rotation amount.
    """
    n, ct, st, rmat = axis_rotation(theta)

    rmat[:, 0, 0:3] = np.array([[1, 0, 0]])
    rmat[:, 1, 0:3] = np.vstack([np.zeros(n), ct, -st]).T
    rmat[:, 2, 0:3] = np.vstack([np.zeros(n), st, ct]).T

    return rmat.squeeze()


def rotation_y_axis(theta):
    """
    Returns a rotation matrix of a rotation about the Y-axis.

    Supports vector-valued theta, in which case the returned array is of shape (len(t), 3, 3), or (len(t), 4, 4) if
    full=True. If theta is scalar the batch dimension is squeezed out.

    Args:
      theta: The rotation amount.
    """
    n, ct, st, rmat = axis_rotation(theta)

    rmat[:, 0, 0:3] = np.vstack([ct, np.zeros(n), st]).T
    rmat[:, 1, 0:3] = np.array([[0, 1, 0]])
    rmat[:, 2, 0:3] = np.vstack([-st, np.zeros(n), ct]).T

    return rmat.squeeze()


def rotation_z_axis(theta):
    """
    Returns a rotation matrix of a rotation about the z-axis.

    Supports vector-valued theta, in which case the returned array is of shape (len(t), 3, 3), or (len(t), 4, 4) if
    full=True. If theta is scalar the batch dimension is squeezed out.

    Args:
      theta: The rotation amount.
    """
    n, ct, st, rmat = axis_rotation(theta)

    rmat[:, 0, 0:3] = np.vstack([ct, -st, np.zeros(n)]).T
    rmat[:, 1, 0:3] = np.vstack([st, ct, np.zeros(n)]).T
    rmat[:, 2, 0:3] = np.array([[0, 0, 1]])

    return rmat.squeeze()


def euler2rmat_xyz(euler_vec: np.ndarray) -> np.ndarray:
    """
    Returns the rotation matrix (or transform) for the given Euler rotations.

    This method composes the Rotation matrix corresponding to the given rotations r1, r2, r3 following the ordering
    "XYZ". To be more precised, the "XYZ" ordering specifies the order of rotation matrices in the matrix
    multiplication, e.g., performing rotX(r1) @ rotY(r2) @ rotZ(r3).

    :param euler_vec: The euler angle rotations.

    :return: The rotation matrix or homogenous transform corresponding to the given Euler rotation (as a matrix)
    """
    rotations = [{"X": rotation_x_axis, "Y": rotation_y_axis, "Z": rotation_z_axis}[c] for c in "XYZ"]
    euler_vec = np.atleast_2d(euler_vec)

    # Get matrices...
    rots = [rotations[i](euler_vec[:, i]) for i in range(len(rotations))]
    return rots[0].dot(rots[1]).dot(rots[2])


def mat2quat(rmat: np.ndarray) -> np.ndarray:
    """
    Return quaternion from homogeneous or rotation matrix.

    Args:
        rmat: A homogeneous transform or rotation matrix
    Returns:
        A quaternion [w, i, j, k].
    """
    if rmat.shape == (3, 3):
        tmp = np.eye(4)
        tmp[0:3, 0:3] = rmat
        rmat = tmp

    q = np.empty((4,), dtype=np.float64)
    t = np.trace(rmat)
    if t > rmat[3, 3]:
        q[0] = t
        q[3] = rmat[1, 0] - rmat[0, 1]
        q[2] = rmat[0, 2] - rmat[2, 0]
        q[1] = rmat[2, 1] - rmat[1, 2]
    else:
        i, j, k = 0, 1, 2
        if rmat[1, 1] > rmat[0, 0]:
            i, j, k = 1, 2, 0
        if rmat[2, 2] > rmat[i, i]:
            i, j, k = 2, 0, 1
        t = rmat[i, i] - (rmat[j, j] + rmat[k, k]) + rmat[3, 3]
        q[i + 1] = t
        q[j + 1] = rmat[i, j] + rmat[j, i]
        q[k + 1] = rmat[k, i] + rmat[i, k]
        q[0] = rmat[k, j] - rmat[j, k]
    q *= 0.5 / np.sqrt(t * rmat[3, 3])
    return q


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
    """
    mat = quat2rmat(quat)
    if ordering == "XYZ":
        return rmat2euler_xyz(mat[0:3, 0:3])
    else:
        raise NotImplementedError("Quat2Euler for Ordering != XYZ is not yet defined!")


def euler2quat(euler_vec: np.ndarray):
    """
    Returns the Quaternion corresponding to the provided euler angles via the strict rotation ordering "XYZ".

    Args:
        euler_vec: The euler angle rotations.

    Returns:
        quat: A quaternion [w, i, j, k]
    """
    rmat = euler2rmat_xyz(euler_vec)
    return mat2quat(rmat)


def rotations() -> None:
    # Helper functions for generating a rotation trajectory to follow (figure-eight)
    def generate_figure_eight(t: Union[float, np.ndarray]) -> np.ndarray:
        x = (np.sin(t) * np.cos(t)).reshape(-1, 1)
        y = np.sin(t).reshape(-1, 1)
        z = (np.cos(t) * np.cos(t)).reshape(-1, 1)
        return np.concatenate([x, y, z], axis=1)

    def figure_eight(t: Union[float, np.ndarray], scale: float = 0.5) -> np.ndarray:
        # Shift curve orientation to start at current gripper orientation
        curve, origin = generate_figure_eight(t) * scale, generate_figure_eight(0.0) * scale
        return curve - origin + EE_ORIENTATION

    # Generate the desired trajectory --> the "gold" path to follow...
    timesteps = np.linspace(0, 2 * np.pi, 50)
    desired = figure_eight(timesteps)

    # Try to "reconstruct" the figure-eight via different rotation conversions to/from quaternions...
    curr_t, max_t, dm_actual, scipy_actual = 0, 2 * np.pi, [], []
    while curr_t < max_t:
        target_euler_angle = figure_eight(curr_t).flatten()

        # DM Control Functions w/ Backtranslation
        dm_desired_quat = euler2quat(target_euler_angle)
        dm_actual_euler = quat2euler(dm_desired_quat)

        # Scipy Rotation w/ Backtranslation
        scipy_desired_quat = R.from_euler("xyz", target_euler_angle).as_quat()
        scipy_actual_euler = R.from_quat(scipy_desired_quat).as_euler("xyz")

        # Log
        dm_actual.append(dm_actual_euler)
        scipy_actual.append(scipy_actual_euler)

        # Update time (assume 20 HZ)
        curr_t += 1.0 / HZ

    # Vectorize
    dm_actual, scipy_actual = np.asarray(dm_actual), np.asarray(scipy_actual)

    # Plot target (black) vs. dm_actual (blue) vs. scipy_actual (red)
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.plot3D(desired[:, 0], desired[:, 1], desired[:, 2], "black", label="Ground Truth")
    ax.scatter3D(dm_actual[:, 0], dm_actual[:, 1], dm_actual[:, 2], c="blue", marker="o", alpha=0.7, label="DM Control")
    ax.scatter3D(
        scipy_actual[:, 0], scipy_actual[:, 1], scipy_actual[:, 2], c="green", marker="^", alpha=0.7, label="Scipy"
    )
    ax.legend()
    ax.set_title("Euler Angles -> Quat -> Euler Angles for DM Control vs. Scipy", fontdict={"fontsize": 18}, pad=25)

    # Save Figure
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/rotations.png")


if __name__ == "__main__":
    rotations()
