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
from typing import Any, Dict, List, Optional, Tuple

import grpc
import gym
import numpy as np
import torch
import torchcontrol as toco
from gym import Env
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R


# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402


# Get Logger
log = logging.getLogger(__name__)

# Silence OpenAI Gym warnings
gym.logger.setLevel(logging.ERROR)

# Control Frequency
HZ = 20

# fmt: off
HOMES = {"default": [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0]}

# Joint Impedance Controller gains...
#   =>> Libfranka Defaults (https://frankaemika.github.io/libfranka/joint_impedance_control_8cpp-example.html)
KQ_GAINS, KQD_GAINS = {"default": [600, 600, 600, 600, 250, 150, 50]}, {"default": [50, 50, 50, 50, 30, 25, 15]}

# End-Effector Impedance Controller gains...
#   =>> P :: Libfranka Defaults (https://frankaemika.github.io/libfranka/cartesian_impedance_control_8cpp-example.html)
#   =>> D :: Libfranka Defaults = int(2 * sqrt(KP))
KX_GAINS, KXD_GAINS = {"default": [150, 150, 150, 10, 10, 10]}, {"default": [25, 25, 25, 7, 7, 7]}

# Resolved Rate Controller Gains =>> should get lower as you get to the end-effector...
KRR_GAINS = {"default": [50, 50, 50, 50, 30, 20, 10]}
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

    def __init__(self, Kp: torch.Tensor, robot_model: torch.nn.Module, ignore_gravity: bool = True) -> None:
        """
        Initializes a Resolved Rates controller with the given P gains and robot model.

        :param Kp: P gains in joint space (7-DoF)
        :param robot_model: A robot model from torchcontrol.models
        :param ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize Modules --> Inverse Dynamics is necessary as it needs to be compensated for in output torques...
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(self.robot_model, ignore_gravity=ignore_gravity)

        # Create LinearFeedback (P) Controller...
        self.p = toco.modules.feedback.LinearFeedback(Kp)

        # Reference End-Effector Velocity (dx, dy, dz, droll, dpitch, dyaw)
        self.ee_velocity_desired = torch.nn.Parameter(torch.zeros(6))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute joint torques given desired EE velocity.

        :param state_dict: A dictionary containing robot states (joint positions, velocities, etc.)
        :return Dictionary containing joint torques.
        """
        # State Extraction
        joint_pos_current, joint_vel_current = state_dict["joint_positions"], state_dict["joint_velocities"]

        # Compute Target Joint Velocity via Resolved Rate Control...
        #   =>> Resolved Rate: joint_vel_desired = J.pinv() @ ee_vel_desired
        #                      >> Numerically stable --> torch.linalg.lstsq(J, ee_vel_desired).solution
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        joint_vel_desired = torch.linalg.lstsq(jacobian, self.ee_velocity_desired).solution

        # Control Logic --> Compute P Torque (feedback) & Inverse Dynamics Torque (feedforward)
        torque_feedback = self.p(joint_vel_current, joint_vel_desired)
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

        :return Tuple of (3D position, 4D orientation as a quaternion w/ scalar-last)
        """
        pos, quat = super().get_ee_pose()
        return pos, torch.roll(quat, -1)

    def start_resolved_rate_control(self, Kq: Optional[List[float]] = None) -> List[Any]:
        """
        Start Resolved-Rate Control (P control on Joint Velocity), as a non-blocking controller.

        The desired EE velocities can be updated using `update_desired_ee_velocities` (6-DoF)!
        """
        torch_policy = ResolvedRateControl(
            Kp=self.Kqd_default if Kq is None else Kq,
            robot_model=self.robot_model,
            ignore_gravity=self.use_grav_comp,
        )
        return self.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def update_desired_ee_velocities(self, ee_velocities: torch.Tensor):
        """
        Update the desired end-effector velocities (6-DoF x, y, z, roll, pitch, yaw).

        Requires starting a resolved-rate controller via `start_resolved_rate_control` beforehand.
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
        :param mode: Mode in < "default" | ... > -- mostly used to set gains!
        :param step_size: Step size to use for `time_to_go` calculations...
        """
        self.home, self.rate, self.mode, self.controller, self.curr_step = home, Rate(hz), mode, controller, 0
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot, self.kp, self.kpd = None, None, None
        self.step_size = step_size

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

        elif self.controller == "resolved-rate":
            # Note: P/D values of "None" default to... well the "default" values for Joint PD Control above 😅
            #   > These values are defined in the default launch_robot YAML (`robot_client/franka_hardware.yaml`)
            self.robot.start_resolved_rate_control(Kq=self.kp)

        else:
            raise NotImplementedError(f"Support for controller `{self.controller}` not yet implemented!")

    def reset(self) -> Dict[str, np.ndarray]:
        # Set PD Gains -- kp, kpd -- depending on current mode, controller
        if self.controller == "joint" and not self.mode == "default":
            self.kp, self.kpd = KQ_GAINS[self.mode], KQD_GAINS[self.mode]
        elif self.controller == "cartesian" and not self.mode == "default":
            self.kp, self.kpd = KX_GAINS[self.mode], KXD_GAINS[self.mode]
        elif self.controller == "resolved-rate":
            self.kp = KRR_GAINS[self.mode]

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
        if self.controller in {"joint", "cartesian", "resolved-rate"}:
            self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot = None, None
        time.sleep(1)


# === Logitech Gamepad/Joystick Controller ===
class JoystickControl:
    def __init__(self, scale: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.5, 0.5, 0.5)) -> None:
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband, self.scale = 0.1, scale

    def input(self) -> Tuple[np.ndarray, bool, bool, bool, bool, bool]:
        pygame.event.get()

        # Reference for the various Joystick `get_axis(i)` --> this is mostly applicable to the Logitech Gamepad...
        #   =>> axis = 0 :: Left Joystick -- right is positive, left is negative (*ignored*)
        #   =>> axis = 1 :: Left Joystick -- up is negative, down is positive (*used for z/yaw control*)
        #   =>> axis = 2 :: Left Trigger -- "off" is negative, "on" is positive (*ignored*)
        #
        #   =>> axis = 3 :: Right Joystick -- right is positive, left is negative (*used for x/roll control*)
        #   =>> axis = 4 :: Right Joystick -- up is negative, down is positive (*used for y/pitch control*)
        #   =>> axis = 5 :: Right Trigger -- "off" is negative, "on" is positive (*used for mode-switching*)

        # Directly compute end-effector velocities from joystick inputs -- switch on right-trigger
        mode = "linear" if self.gamepad.get_axis(5) < 0 else "angular"
        ee_dot = np.zeros(6)

        # Iterate through three axes (x/roll, y/pitch, z/yaw) --> in that order (flipping signs for the latter two axes)
        if mode == "linear":
            x, y, z = self.gamepad.get_axis(3), -self.gamepad.get_axis(4), -self.gamepad.get_axis(1)
            ee_dot[:3] = [vel * self.scale[i] if abs(vel) >= self.deadband else 0 for i, vel in enumerate([x, y, z])]
        else:
            r, p, y = self.gamepad.get_axis(3), -self.gamepad.get_axis(4), -self.gamepad.get_axis(1)
            ee_dot[3:] = [vel * self.scale[i + 3] if abs(vel) >= self.deadband else 0 for i, vel in enumerate([r, p, y])]

        # Button Press
        a, b = self.gamepad.get_button(0), self.gamepad.get_button(1)
        x, y, stop = self.gamepad.get_button(2), self.gamepad.get_button(3), self.gamepad.get_button(7)
        return ee_dot, a, b, x, y, stop


def marionette() -> None:
    """Run 6-DoF Teleoperation w/ a Joystick --> 2 modes, 3 Joystick axes :: (x, y, z) || (roll, pitch, yaw)."""
    cfg = {
        "id": "default-resolved-rate",
        "home": "default",
        "hz": HZ,
        "controller": "resolved-rate",
        "mode": "default",
        "step_size": 0.05,
    }
    print(f"[*] Attempting teleoperation with motion controller `{cfg['controller']}` and `{cfg['id']}` config:")
    for key in cfg:
        print(f"\t`{key}` =>> `{cfg[key]}`")

    # Initialize Joystick...
    print("[*] Connecting to Joystick...")
    joystick = JoystickControl()

    # Initialize environment & get initial poses...
    print("[*] Initializing robot connection...")
    env = FrankaEnv(
        home=cfg["home"], hz=cfg["hz"], controller=cfg["controller"], mode=cfg["mode"], step_size=cfg["step_size"]
    )

    print("[*] Dropping into Teleoperation Loop...")
    try:
        while True:
            # Measure Joystick Input
            endeff_velocities, _, _, _, _, stop = joystick.input()

            if stop:
                # Gracefully exit...
                break

            # Otherwise, drop into velocity control...
            env.step(endeff_velocities)

    except KeyboardInterrupt:
        # Just don't crash the program on Ctrl-C or Socket Error (Controller Death)
        print("\n[*] Terminating...")

    finally:
        env.close()


if __name__ == "__main__":
    marionette()
