"""
puppetry.py

Standalone script for 6-DoF (joystick-based) teleoperated control of a Franka Emika Panda Arm, using Polymetis.
Additionally supports functionality for testing trajectory following and visualization, as well as hooks for other
control interfaces (e.g., a SpaceMouse or Oculus/VR-based controller).

As we're using Polymetis, you should use the following to launch the robot controller:
    > launch_robot.py --config-path /home/iliad/Projects/oncorr/conf/robot --config-name robot_launch.yaml timeout=15;
"""
import logging
import os
import time
from typing import Dict, Optional, Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import Env
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R


# Silence OpenAI Gym warnings
gym.logger.setLevel(logging.ERROR)

# Whether to run in "teleoperation" or "follow" mode (follow is for benchmarking EE controller)
RUN_MODE = "follow"

# What `Rotation` implementation to use in < dm | scipy >
ROTATION_IMPLEMENTATION = "dm"

# fmt: off
HOMES = {
    "default": [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],

    # >> Home Position from @Sasha's Code (IRIS)
    "iris": [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],

    # >> Home Positions for the RB2 Tasks... ignore!
    # "pour": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    # "scoop": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
    # "zip": [-0.1337, 0.3634, -0.1395, -2.3153, 0.1478, 2.7733, -1.1784],
    # "insertion": [0.1828, -0.4909, -0.0093, -2.4412, 0.2554, 3.3310, 0.5905],
}
# fmt: on

# Control Frequency & other useful constants...
HZ, POLE_LIMIT, TOLERANCE = 20, (1.0 - 1e-6), 1e-10

# Joint Controller gains -- we want a compliant robot when recording, and stiff when playing back / teleoperating
KQ_GAINS = {
    "default": [80, 120, 100, 100, 70, 50, 20],
}
KQD_GAINS = {
    "default": [10, 10, 10, 10, 5, 5, 5],
}

# End-Effector Controller gains -- we want a compliant robot when recording, and stiff when playing back / operating
KX_GAINS = {
    "default": [150, 150, 150, 10, 10, 10],
    "teleoperate": [x * 10 for x in [150, 150, 150, 10, 10, 10]],
}
KXD_GAINS = {
    "default": [25, 25, 25, 7, 7, 7],
    "teleoperate": [0, 0, 0, 0, 0, 0],  # We want straight up linear feedback!
}


# Hardcoded Low/High Joint Thresholds for the Franka Emika Panda Arm
LOW_JOINTS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HIGH_JOINTS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the POMDP loop.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()


# === DM Control Rotation & Quaternion Helpers ===
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


# === Polymetis Environment Wrapper ===
class FrankaEnv(Env):
    def __init__(
        self, home: str, hz: int, mode: str = "default", controller: str = "cartesian", step_size: float = 0.05
    ) -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param home: Default home position (specified as a string index into `HOMES` above!
        :param hz: Default policy control Hz; somewhere between 20-60 is a good range.
        :param mode: Mode in < "record" | "default" | "teleoperate"> -- mostly used to set gains!
        :param controller: Which impedance controller to use in < joint | cartesian | osc > (teleoperate uses osc!)
        :param step_size: Step size to use for `time_to_go` calculations...
        """
        self.home, self.rate, self.mode, self.controller, self.curr_step = home, Rate(hz), mode, controller, 0
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot, self.kp, self.kpd = None, None, None
        self.step_size = step_size

        # Debugging...
        self.current_ee_rot_dm, self.current_ee_rot_scipy = None, None

        # Initialize Robot and PD Controller
        self.reset()

    def robot_setup(self, home: str, franka_ip: str = "172.16.0.1") -> None:
        # Initialize Robot Interface and Reset to Home
        self.robot = RobotInterface(ip_address=franka_ip)
        self.robot.set_home_pose(torch.Tensor(HOMES[home]))
        self.robot.go_home()

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = quat2euler(self.current_ee_pose[3:])
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

        # Debugging...
        self.current_ee_rot_dm = quat2euler(self.current_ee_pose[3:])
        self.current_ee_rot_scipy = R.from_quat(self.current_ee_pose[3:]).as_euler("xyz")

        # Create an *Impedance Controller*, with the desired gains...
        #   > Note: Feel free to add any other controller, e.g., a PD controller around joint poses.
        #           =>> Ref: https://github.com/AGI-Labs/franka_control/blob/master/util.py#L74
        if self.controller == "joint":
            # Note: P/D values of "None" default to... well the "default" values above 😅
            #   > These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
            self.robot.start_joint_impedance(Kq=self.kp, Kqd=self.kpd)

        elif self.controller == "cartesian":
            # Note: P/D values of "None" default to... well the "default" values above 😅
            #   > These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
            self.robot.start_cartesian_impedance(Kx=self.kp, Kxd=self.kpd)

        elif self.controller == "osc":
            # Don't think you actually need to start a new controller here... should use the Polymetis backoff?
            pass

        else:
            raise NotImplementedError(f"Support for controller `{self.controller}` not yet implemented!")

    def reset(self) -> Dict[str, np.ndarray]:
        # Set PD Gains -- kp, kpd -- depending on current mode, controller
        if self.controller == "joint" and not self.mode == "default":
            self.kp, self.kpd = KQ_GAINS[self.mode], KQD_GAINS[self.mode]
        elif self.controller == "cartesian" and not self.mode == "default":
            self.kp, self.kpd = KX_GAINS[self.mode], KXD_GAINS[self.mode]

        # Call setup with the new controller...
        self.robot_setup(self.home)
        return self.get_obs()

    def get_obs(self) -> Dict[str, np.ndarray]:
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = quat2euler(new_ee_pose[3:])

        # Debugging...
        new_ee_rot_dm = quat2euler(new_ee_pose[3:])
        new_ee_rot_scipy = R.from_quat(new_ee_pose[3:]).as_euler("xyz")

        # Note that deltas are "shifted" 1 time step to the right from the corresponding "state"
        obs = {
            "q": new_joint_pose,
            "qdot": self.robot.get_joint_velocities().numpy(),
            "delta_q": new_joint_pose - self.current_joint_pose,
            "ee_pose": new_ee_pose,
            "delta_ee_pose": new_ee_pose - self.current_ee_pose,
        }

        # Bump "current" trackers...
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = new_joint_pose, new_ee_pose, new_ee_rot

        # Debugging...
        self.current_ee_rot_dm, self.current_ee_rot_scipy = new_ee_rot_dm, new_ee_rot_scipy

        return obs

    def step(self, action: Optional[np.ndarray]) -> Tuple[Dict[str, np.ndarray], int, bool, None]:
        """Run a step in the environment, where `delta` specifies if we are sending absolute poses or deltas in poses!"""
        if action is not None:
            if self.controller == "joint":
                # Joint Impedance Controller expects 7D Joint Angles
                q = torch.from_numpy(action)
                self.robot.update_desired_joint_positions(q)

            elif self.controller == "cartesian":
                # Cartesian controller expects tuple -- first 3 elements are xyz, last 4 are quaternion orientation...
                pos, ori = torch.from_numpy(action[:3]), torch.from_numpy(action[3:])
                self.robot.update_desired_ee_pose(position=pos, orientation=ori)

            elif self.controller == "osc":
                # OSC controller expects tuple -- first 3 elements are xyz, last 4 are quaternion orientation...
                #   =>> Note: `move_to_ee_pose` does not natively accept Tensors!
                pos, ori = action[:3], action[3:]
                self.robot.move_to_ee_pose(position=pos, orientation=ori, time_to_go=self.step_size)
            else:
                raise NotImplementedError(f"Controller mode `{self.controller}` not supported!")

        # Sleep according to control frequency
        self.rate.sleep()

        # Return observation, Gym default signature...
        return self.get_obs(), 0, False, None

    @property
    def ee_position(self) -> np.ndarray:
        """Return current EE position --> 3D x/y/z!"""
        return self.current_ee_pose[:3]

    @property
    def ee_orientation(self) -> np.ndarray:
        """Return current EE orientation as euler angles (in radians) --> 3D roll/pitch/yaw!"""
        return self.current_ee_rot

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("Render is not implemented for Physical FrankaEnv...")

    def close(self) -> None:
        # Terminate Policy
        if self.controller in {"joint", "cartesian"}:
            self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot = None, None
        time.sleep(1)


def teleoperate() -> None:
    """Run 6-DoF Teleoperation w/ a Joystick --> 2 modes, 3 Joystick axes :: (x, y, z) || (roll, pitch, yaw)."""
    raise NotImplementedError("Not implemented until trajectory following works...")


def follow() -> None:
    """Follow a 3D figure-eight trajectory with the current EE controller, plotting expected vs. actual trajectories."""

    # === Define a few plausible configurations ===
    # cfg = {
    #     "id": "default-cartesian",
    #     "home": "iris",
    #     "hz": HZ,
    #     "mode": "default",
    #     "controller": "cartesian",
    #     "step_size": 0.05,
    # }
    # cfg = {
    #     "id": "default-move-to-ee",
    #     "home": "iris",
    #     "hz": HZ,
    #     "mode": "default",
    #     "controller": "osc",
    #     "step_size": 0.05,
    # }
    cfg = {
        "id": "linear-feedback",
        "home": "iris",
        "hz": HZ,
        "mode": "teleoperate",
        "controller": "cartesian",
        "step_size": 0.05,
    }
    print(f"[*] Attempting to perform trajectory following with EE impedance controller and `{cfg['id']}` config:")
    for key in cfg:
        print(f"\t`{key}` =>> `{cfg[key]}`")

    # Initialize environment & get initial poses...
    os.makedirs("plots", exist_ok=True)
    env = FrankaEnv(
        home=cfg["home"], hz=cfg["hz"], mode=cfg["mode"], controller=cfg["controller"], step_size=cfg["step_size"]
    )
    fixed_position, ee_orientation = env.ee_position, env.ee_orientation

    # Helper functions for generating a rotation trajectory to follow (figure-eight)
    def generate_figure_eight(t: Union[float, np.ndarray]) -> np.ndarray:
        x = (np.sin(t) * np.cos(t)).reshape(-1, 1)
        y = np.sin(t).reshape(-1, 1)
        z = (np.cos(t) * np.cos(t)).reshape(-1, 1)
        return np.concatenate([x, y, z], axis=1)

    def figure_eight(t: Union[float, np.ndarray], scale: float = 0.5) -> np.ndarray:
        # Shift curve orientation to start at current gripper orientation
        curve, origin = generate_figure_eight(t) * scale, generate_figure_eight(0.0) * scale
        return curve - origin + ee_orientation

    # Generate the desired trajectory --> the "gold" path to follow...
    timesteps = np.linspace(0, 2 * np.pi, 50)
    desired = figure_eight(timesteps)

    # Drop into follow loop --> we're just tracing a figure eight with the orientation (fixed position!)
    curr_t, max_t, actual, deltas = 0, 2 * np.pi, [ee_orientation], [figure_eight(0.0).flatten() - ee_orientation]
    actual_dm, actual_scipy = [], []

    # Wrap in try/except...
    try:
        while curr_t < max_t:
            # Move Robot --> transform Euler angle back to quaternion...
            new_angle = figure_eight(curr_t).flatten()

            if ROTATION_IMPLEMENTATION == "dm":
                new_quat = euler2quat(new_angle)
            elif ROTATION_IMPLEMENTATION == "scipy":
                new_quat = R.from_euler("xyz", new_angle).as_quat()
            else:
                raise NotImplementedError(f"Rotation Implementation `{ROTATION_IMPLEMENTATION}` not found!")

            # Take a step...
            env.step(np.concatenate([fixed_position, new_quat], axis=0))

            # Grab updated orientation
            achieved_orientation = env.ee_orientation
            actual.append(achieved_orientation)
            deltas.append(new_angle - achieved_orientation)

            # Debugging...
            actual_dm.append(env.current_ee_rot_dm)
            actual_scipy.append(env.current_ee_rot_scipy)

            # Update Time
            print(f"Target: {new_angle} -- Achieved: {achieved_orientation}")
            curr_t += cfg["step_size"]

    except KeyboardInterrupt:
        print("[*] Keyboard Interrupt - Robot in Unsafe State...")

    # Vectorize Trajectory
    actual = np.asarray(actual)
    actual_dm, actual_scipy = np.asarray(actual_dm), np.asarray(actual_scipy)

    # Plot desired (black) vs. actual (red)
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.plot3D(desired[:, 0], desired[:, 1], desired[:, 2], "black", label="Ground Truth")
    ax.scatter3D(actual[:, 0], actual[:, 1], actual[:, 2], c="red", alpha=0.7, label="Actual")
    plt.savefig(f"plots/{cfg['id']}+m={cfg['mode']}+r={ROTATION_IMPLEMENTATION}.png")
    ax.legend()
    ax.set_title("Desired vs. Actual Robot Trajectory", fontdict={"fontsize": 18}, pad=25)
    plt.savefig(f"plots/trajectory-{cfg['id']}+m={cfg['mode']}+r={ROTATION_IMPLEMENTATION}.png")
    plt.clf()

    # Plot desired (black) vs. DM (blue) vs. Scipy (green)
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.plot3D(desired[:, 0], desired[:, 1], desired[:, 2], "black", label="Ground Truth")
    ax.scatter3D(actual_dm[:, 0], actual_dm[:, 1], actual_dm[:, 2], c="blue", alpha=0.7, label="DM Control")
    ax.scatter3D(actual_scipy[:, 0], actual_scipy[:, 1], actual_scipy[:, 2], c="green", alpha=0.7, label="Scipy")
    ax.legend()
    ax.set_title("Euler Angles -> Quat -> Euler Angles for DM Control vs. Scipy", fontdict={"fontsize": 18}, pad=25)
    plt.savefig(f"plots/{cfg['id']}+m={cfg['mode']}+r={ROTATION_IMPLEMENTATION}.png")
    plt.clf()

    # Cleanup
    env.close()

    # Explore...
    norms = np.array([np.linalg.norm(x) for x in deltas])
    print(f"Maximum Euler Angle Difference (L2 Norm) = `{norms.max()}`")

    # fmt: off
    if input("Drop into (p)rompt? [Press any other key to exit] =>> ") == "p":
        import IPython
        IPython.embed()

    print("[*] Exiting...")
    # fmt: on


if __name__ == "__main__":
    if RUN_MODE == "teleoperate":
        teleoperate()
    elif RUN_MODE == "follow":
        follow()
    else:
        print(f"Run mode `{RUN_MODE}` not implemented -- try one of < teleoperate | follow >")
