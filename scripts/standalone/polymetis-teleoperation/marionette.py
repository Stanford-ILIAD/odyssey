"""
marionette.py

Standalone script for 6-DoF (joystick-based) teleoperated control of a Franka Emika Panda Arm, using Polymetis.
Additionally supports functionality for testing trajectory following and visualization, as well as hooks for other
control interfaces (e.g., a SpaceMouse or Oculus/VR-based controller).

As we're using Polymetis, you should use the following to launch the robot controller:
    > launch_robot.py --config-path /home/iliad/Projects/oncorr/conf/robot --config-name robot_launch.yaml timeout=15;
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import grpc
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchcontrol as toco
from gym import Env
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R


# Get Logger
log = logging.getLogger(__name__)

# Silence OpenAI Gym warnings
gym.logger.setLevel(logging.ERROR)

# Whether to run in "teleoperation" or "follow" mode (follow is for benchmarking EE controller)
RUN_MODE = "follow"

# Control Frequency
HZ = 20

# fmt: off
HOMES = {
    "default": [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],

    # >> Home Position for ILIAD (same as default)
    "iliad": [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0],

    # >> Home Position from @Sasha's Code (IRIS)
    "iris": [0.0, 0.0, 0.0, -np.pi / 2.0, 0.0, np.pi / 2.0, 0.0],
}

# Joint Controller gains -- we want a compliant robot when recording, and stiff when playing back / teleoperating
#   =>> Libfranka Defaults (https://frankaemika.github.io/libfranka/joint_impedance_control_8cpp-example.html)
#       - `k_gains` and `d_gains` are both hardcoded...
KQ_GAINS = {
    "default": [600, 600, 600, 600, 250, 150, 50],
    "teleoperate": [600, 600, 600, 600, 250, 150, 50],
}
KQD_GAINS = {
    "default": [50, 50, 50, 50, 30, 25, 15],
    "teleoperate": [0, 0, 0, 0, 0, 0, 0],   # We want straight up linear feedback!
}

# End-Effector Controller gains -- we want a compliant robot when recording, and stiff when playing back / operating
#   =>> Libfranka Defaults (https://frankaemika.github.io/libfranka/cartesian_impedance_control_8cpp-example.html)
KX_GAINS = {
    "default": [150, 150, 150, 10, 10, 10],
    "teleoperate": [150, 150, 150, 10, 10, 10],
}
#   =>> Libfranka Defaults = int(2 * sqrt(KP))
KXD_GAINS = {
    "default": [25, 25, 25, 7, 7, 7],
    "teleoperate": [0, 0, 0, 0, 0, 0],  # We want straight up linear feedback!
}
# fmt: on


class Rate:
    def __init__(self, frequency: float) -> None:
        """
        Maintains a constant control rate for the control loop.

        :param frequency: Polling frequency, in Hz.
        """
        self.period, self.last = 1.0 / frequency, time.time()

    def sleep(self) -> None:
        current_delta = time.time() - self.last
        sleep_time = max(0.0, self.period - current_delta)
        if sleep_time:
            time.sleep(sleep_time)
        self.last = time.time()


# === Resolved Rates Controller===
class ResolvedRateControl(toco.PolicyModule):
    """Resolved Rates Control --> End-Effector Control (dx, dy, dz, droll, dpitch, dyaw) via Joint Velocity Control"""

    def __init__(self, Kp, Kd, robot_model: torch.nn.Module, hz: int, ignore_gravity: bool = True) -> None:
        """
        Initializes a Resolved Rates controller with the given PD gains and robot model.

        :param Kp: P gains in joint space (7-DoF)
        :param Kd: D gains in joint space (7-DoF)
        :param robot_model: A robot model from torchcontrol.models
        :param hz: Frequency of the calls to forward()
        :param ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize Modules --> Inverse Dynamics is necessary as it needs to be compensated for in output torques...
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(self.robot_model, ignore_gravity=ignore_gravity)
        self.hz, self.dt, self.is_initialized = hz, 1.0 / hz, False

        # Create JointPD Controller...
        self.pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Reference End-Effector Velocity (dx, dy, dz, droll, dpitch, dyaw)
        self.ee_velocity_desired = torch.nn.Parameter(torch.zeros(6))
        self.joint_pos_desired = torch.zeros(7)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute joint torques given desired EE velocity.

        :param state_dict: A dictionary containing robot states (joint positions, velocities, etc.)
        :return Dictionary containing joint torques.
        """
        # State Extraction
        joint_pos_current, joint_vel_current = state_dict["joint_positions"], state_dict["joint_velocities"]
        if not self.is_initialized:
            self.joint_pos_desired, self.is_initialized = torch.clone(joint_pos_current), True

        # Compute Target Joint Velocity via Resolved Rate Control...
        #   =>> Resolved Rate: joint_vel_desired = J.pinv() @ ee_vel_desired
        #                      >> Numerically stable --> torch.linalg.lstsq(J, ee_vel_desired).solution
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        joint_vel_desired = torch.linalg.lstsq(jacobian, self.ee_velocity_desired).solution

        # Compute new "desired" joint pose for PD control...
        self.joint_pos_desired += torch.mul(joint_vel_desired, self.dt)

        # Control Logic --> Compute PD Torque (feedback) & Inverse Dynamics Torque (feedforward)
        torque_feedback = self.pd(joint_pos_current, joint_vel_current, self.joint_pos_desired, joint_vel_desired)
        torque_feedforward = self.invdyn(joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current))
        torque_out = torque_feedback + torque_feedforward
        return {"joint_torques": torque_out}


# === Odyssey Robot Interface (ensures consistency...)
class OdysseyRobotInterface(RobotInterface):
    def get_ee_pose(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Polymetis defaults to returning a Tuple of (position, orientation), where orientation is a quaternion in
        *scalar-first* format (w, x, y, z). However, `scipy` and other libraries expect *scalar-last* (x, y, z, w);
        we take care of that here!

        Returns: Tuple of (3D position, 4D orientation as a quaternion w/ scalar-last)
        """
        pos, quat = super().get_ee_pose()
        return pos, torch.roll(quat, -1)

    def start_resolved_rate_control(
        self, hz: Optional[int] = None, Kq: Optional[List[float]] = None, Kqd: Optional[List[float]] = None
    ) -> List[Any]:
        """
        Start Resolved-Rate Control (via Joint Velocity PD Control), as a non-blocking controller; the desired EE
        velocities can be updated using `update_desired_ee_velocities` (6-DoF).
        """
        torch_policy = ResolvedRateControl(
            Kp=self.Kq_default if Kq is None else Kq,
            Kd=self.Kqd_default if Kqd is None else Kqd,
            robot_model=self.robot_model,
            hz=self.metadata.hz if hz is None else hz,
            ignore_gravity=self.use_grav_comp,
        )
        return self.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def update_desired_ee_velocities(self, ee_velocities: torch.Tensor):
        """
        Update the desired end-effector velocities (6-DoF x, y, z, roll, pitch, yaw). Requires starting a resolved-rate
        controller via `start_resolved_rate_control` beforehand.
        """
        try:
            update_idx = self.update_current_policy({"ee_velocity_desired": ee_velocities})
        except grpc.RpcError as e:
            log.error(
                "Unable to update desired end-effector velocities. Use `start_resolved_rate_control` to start a "
                "resolved-rate controller."
            )
            raise e
        return update_idx


# === Polymetis Environment Wrapper ===
class FrankaEnv(Env):
    def __init__(
        self, home: str, hz: int, controller: str = "cartesian", mode: str = "default", step_size: float = 0.05
    ) -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param home: Default home position (specified as a string index into `HOMES` above!
        :param hz: Default policy control hz; somewhere between 20-40 is a good range.
        :param controller: Which impedance controller to use in < joint | cartesian | osc >
        :param mode: Mode in < "record" | "default" | "teleoperate"> -- mostly used to set gains!
        :param step_size: Step size to use for `time_to_go` calculations...
        """
        self.home, self.rate, self.mode, self.controller, self.curr_step = home, Rate(hz), mode, controller, 0
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot, self.kp, self.kpd = None, None, None
        self.hz, self.step_size = hz, step_size

        # Initialize Robot and PD Controller
        self.reset()

    def robot_setup(self, home: str, franka_ip: str = "172.16.0.1") -> None:
        # Initialize Robot Interface and Reset to Home
        self.robot = OdysseyRobotInterface(ip_address=franka_ip)
        self.robot.set_home_pose(torch.Tensor(HOMES[home]))
        self.robot.go_home()

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = R.from_quat(self.current_ee_pose[3:]).as_euler("xyz")
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

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

        elif self.controller == "resolved-rate":
            # Note: P/D values of "None" default to... well the "default" values for Joint PD Control above 😅
            #   > These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
            self.robot.start_resolved_rate_control(hz=self.hz, Kq=self.kp, Kqd=self.kpd)

        else:
            raise NotImplementedError(f"Support for controller `{self.controller}` not yet implemented!")

    def reset(self) -> Dict[str, np.ndarray]:
        # Set PD Gains -- kp, kpd -- depending on current mode, controller
        if self.controller in {"joint", "resolved-rate"} and not self.mode == "default":
            self.kp, self.kpd = KQ_GAINS[self.mode], KQD_GAINS[self.mode]
        elif self.controller == "cartesian" and not self.mode == "default":
            self.kp, self.kpd = KX_GAINS[self.mode], KXD_GAINS[self.mode]

        # Call setup with the new controller...
        self.robot_setup(self.home)
        return self.get_obs()

    def get_obs(self) -> Dict[str, np.ndarray]:
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = R.from_quat(new_ee_pose[3:]).as_euler("xyz")

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

            elif self.controller == "resolved-rate":
                # Resolved rate controller expects 6D end-effector velocities (deltas) in X/Y/Z/Roll/Pitch/Yaw...
                ee_velocities = torch.from_numpy(action)
                self.robot.update_desired_ee_velocities(ee_velocities)

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
    #     "id": "default-cartesian-impedance",
    #     "home": "iris",
    #     "hz": HZ,
    #     "controller": "cartesian",
    #     "mode": "default",
    #     "step_size": 0.05,
    # }
    # cfg = {
    #     "id": "cartesian-linear-feedback",
    #     "home": "iris",
    #     "hz": HZ,
    #     "controller": "cartesian",
    #     "mode": "teleoperate",
    #     "step_size": 0.05,
    # }
    cfg = {
        "id": "default-resolved-rate",
        "home": "iris",
        "hz": HZ,
        "controller": "resolved-rate",
        "mode": "default",
        "step_size": 0.05,
    }
    print(f"[*] Attempting trajectory following with controller `{cfg['controller']}` and `{cfg['id']}` config:")
    for key in cfg:
        print(f"\t`{key}` =>> `{cfg[key]}`")

    # Initialize environment & get initial poses...
    env = FrankaEnv(
        home=cfg["home"], hz=cfg["hz"], controller=cfg["controller"], mode=cfg["mode"], step_size=cfg["step_size"]
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
    curr_t, max_t, actual = cfg["step_size"], 2 * np.pi, [ee_orientation]
    achieved_orientation, deltas = env.ee_orientation, [figure_eight(0.0).flatten() - ee_orientation]

    # Wrap in try/except...
    try:
        while curr_t < max_t:
            # Move Robot --> transform Euler angle back to quaternion...
            new_angle = figure_eight(curr_t).flatten()
            new_quat = R.from_euler("xyz", new_angle).as_quat()

            # Take a step...
            if cfg["controller"] in {"cartesian", "osc"}:
                env.step(np.concatenate([fixed_position, new_quat], axis=0))
            elif cfg["controller"] in {"resolved-rate"}:
                env.step(np.concatenate([fixed_position, new_angle - achieved_orientation]))

            # Grab updated orientation
            achieved_orientation = env.ee_orientation
            actual.append(achieved_orientation)
            deltas.append(new_angle - achieved_orientation)

            # Update Time
            print(f"Target: {new_angle} -- Achieved: {achieved_orientation}")
            curr_t += cfg["step_size"]

    except KeyboardInterrupt:
        print("[*] Keyboard Interrupt - Robot in Unsafe State...")

    # Vectorize Trajectory
    actual = np.asarray(actual)

    # Plot desired (black) vs. actual (red)
    os.makedirs("plots/marionette", exist_ok=True)
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.plot3D(desired[:, 0], desired[:, 1], desired[:, 2], "black", label="Ground Truth")
    ax.scatter3D(actual[:, 0], actual[:, 1], actual[:, 2], c="red", alpha=0.7, label="Actual")
    plt.savefig(f"plots/marionette/{cfg['id']}+m={cfg['mode']}.png")
    ax.legend()
    ax.set_title("Desired vs. Actual Robot Trajectory", fontdict={"fontsize": 18}, pad=25)
    plt.savefig(f"plots/trajectory-{cfg['id']}+c={cfg['controller']}+m={cfg['mode']}.png")
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
