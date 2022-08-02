"""
demonstrate.py

Collect interleaved demonstrations (in the case of kinesthetic teaching) of recording a kinesthetic demo,
then (optionally) playing back the demonstration to collect visual states.

As we're using Polymetis, you should use the following to launch the robot controller:
    > launch_robot.py --config-path /home/iliad/Projects/oncorr/conf/robot --config-name robot_launch.yaml timeout=15;
    > launch_gripper.py --config-path /home/iliad/Projects/oncorr/conf/robot --config-name gripper_launch.yaml;

References:
    - https://github.com/facebookresearch/fairo/blob/main/polymetis/examples/2_impedance_control.py
    - https://github.com/AGI-Labs/franka_control/blob/master/record.py
    - https://github.com/AGI-Labs/franka_control/blob/master/playback.py
"""
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import grpc
import gym
import numpy as np
import torch
import torchcontrol as toco
from gym import Env
from polymetis import GripperInterface, RobotInterface
from scipy.spatial.transform import Rotation as R
from tap import Tap


# Suppress PyGame Import Text
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame  # noqa: E402


# Get Logger
overwatch = logging.getLogger(__name__)

# Silence OpenAI Gym warnings
gym.logger.setLevel(logging.ERROR)

# Control Frequency
HZ = 20

# fmt: off
HOMES = {"default": [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0]}

# Control Frequency & other useful constants...
#   > Ref: Gripper constants from: https://frankaemika.github.io/libfranka/grasp_object_8cpp-example.html
GRIPPER_SPEED, GRIPPER_FORCE, GRIPPER_MAX_WIDTH = 0.5, 120, 0.08570

# Joint Impedance Controller gains (used mostly for recording kinesthetic demos & playback)
#   =>> Libfranka Defaults (https://frankaemika.github.io/libfranka/joint_impedance_control_8cpp-example.html)
KQ_GAINS = {
    "record": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "default": [600, 600, 600, 600, 250, 150, 50],
}
KQD_GAINS = {
    "record": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "default": [50, 50, 50, 50, 30, 25, 15],
}

# End-Effector Impedance Controller gains (known to be not great...)
#   Issue Ref: https://github.com/facebookresearch/fairo/issues/1280#issuecomment-1182727019)
#   =>> P :: Libfranka Defaults (https://frankaemika.github.io/libfranka/cartesian_impedance_control_8cpp-example.html)
#   =>> D :: Libfranka Defaults = int(2 * sqrt(KP))
KX_GAINS = {"default": [150, 150, 150, 10, 10, 10], "teleoperate": [200, 200, 200, 10, 10, 10]}
KXD_GAINS = {"default": [25, 25, 25, 7, 7, 7], "teleoperate": [50, 50, 50, 7, 7, 7]}

# Resolved Rate Controller Gains =>> (should be default EE controller...)
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
            overwatch.error(
                "Unable to update desired end-effector velocities. Use `start_resolved_rate_control` to start a "
                "resolved-rate controller."
            )
            raise e
        return update_idx


# === Polymetis Environment Wrapper ===
class FrankaEnv(Env):
    def __init__(
        self,
        home: str,
        hz: int,
        controller: str = "cartesian",
        mode: str = "default",
        use_camera: bool = False,
        use_gripper: bool = False,
    ) -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param home: Default home position (specified as a string index into `HOMES` above!
        :param hz: Default policy control hz; somewhere between 20-40 is a good range.
        :param controller: Which impedance controller to use in < joint | cartesian | osc >
        :param mode: Mode in < "default" | ... > --  used to set P(D) gains!
        :param use_camera: Boolean whether to initialize the Camera connection for recording visual states (WIP)
        :param use_gripper: Boolean whether to initialize the Gripper controller (default: False)
        """
        self.home, self.rate, self.mode, self.controller, self.curr_step = home, Rate(hz), mode, controller, 0
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot, self.kp, self.kpd = None, None, None
        self.use_gripper, self.gripper, self.current_gripper_state, self.gripper_is_open = use_gripper, None, None, True

        # TODO(siddk) :: Add camera support...
        if use_camera:
            raise NotImplementedError("Camera support not yet implemented!")

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

        # Initialize Gripper Interface and Open
        if self.use_gripper:
            self.gripper = GripperInterface(ip_address=franka_ip)
            self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            gripper_state = self.gripper.get_state()
            self.current_gripper_state = {"width": gripper_state.width, "max_width": gripper_state.max_width}
            self.gripper_is_open = True

    def reset(self) -> Dict[str, np.ndarray]:
        # Set PD Gains -- kp, kpd -- depending on current mode, controller
        if self.controller == "joint":
            self.kp, self.kpd = KQ_GAINS[self.mode], KQD_GAINS[self.mode]
        elif self.controller == "cartesian":
            self.kp, self.kpd = KX_GAINS[self.mode], KXD_GAINS[self.mode]
        elif self.controller == "resolved-rate":
            self.kp = KRR_GAINS[self.mode]

        # Call setup with the new controller...
        self.robot_setup(self.home)
        return self.get_obs()

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    def get_obs(self) -> Dict[str, np.ndarray]:
        new_joint_pose = self.robot.get_joint_positions().numpy()
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = R.from_quat(new_ee_pose[3:]).as_euler("xyz")

        if self.use_gripper:
            new_gripper_state = self.gripper.get_state()

            # Note that deltas are "shifted" 1 time step to the right from the corresponding "state"
            obs = {
                "q": new_joint_pose,
                "qdot": self.robot.get_joint_velocities().numpy(),
                "delta_q": new_joint_pose - self.current_joint_pose,
                "ee_pose": new_ee_pose,
                "delta_ee_pose": new_ee_pose - self.current_ee_pose,
                "gripper_width": new_gripper_state.width,
                "gripper_max_width": new_gripper_state.max_width,
                "gripper_open": self.gripper_is_open,
            }

            # Bump "current" poses...
            self.current_joint_pose, self.current_ee_pose = new_joint_pose, new_ee_pose
            self.current_gripper_state = {"width": new_gripper_state.width, "max_width": new_gripper_state.max_width}
            return obs

        else:
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

    def step(
        self, action: Optional[np.ndarray], delta: bool = False, open_gripper: Optional[bool] = None
    ) -> Tuple[Dict[str, np.ndarray], int, bool, None]:
        """Run a step in the environment, where `delta` specifies if we are sending absolute poses or deltas in poses!"""
        if action is not None:
            if self.controller == "joint":
                if not delta:
                    # Joint Impedance Controller expects 7D Joint Angles
                    q = torch.from_numpy(action)
                    self.robot.update_desired_joint_positions(q)
                else:
                    raise NotImplementedError("Delta control for Joint Impedance Controller not yet implemented!")

            elif self.controller == "cartesian":
                if not delta:
                    # Cartesian controller expects tuple -- first 3 elements are xyz, last 4 are quaternion orientation
                    pos, quat = torch.from_numpy(action[:3]), torch.from_numpy(action[3:])
                    self.robot.update_desired_ee_pose(position=pos, orientation=quat)
                else:
                    # Convert from 6-DoF (x, y, z, roll, pitch, yaw) if necessary...
                    assert len(action) == 6, "Delta Control for Cartesian Impedance only supported for Euler Angles!"
                    pos, angle = torch.from_numpy(self.ee_position + action[:3]), self.ee_orientation + action[3:]

                    # Convert angle =>> quaternion (Polymetis expects scalar first, so roll...)
                    quat = torch.from_numpy(np.roll(R.from_euler("xyz", angle).as_quat(), 1))
                    self.robot.update_desired_ee_pose(position=pos, orientation=quat)

            elif self.controller == "resolved-rate":
                # Resolved rate controller expects 6D end-effector velocities (deltas) in X/Y/Z/Roll/Pitch/Yaw...
                ee_velocities = torch.from_numpy(action)
                self.robot.update_desired_ee_velocities(ee_velocities)

            else:
                raise NotImplementedError(f"Controller mode `{self.controller}` not supported!")

        # Gripper Handling...
        if open_gripper is not None and (self.gripper_is_open ^ open_gripper):
            # True --> Open Gripper, otherwise --> Close Gripper
            self.gripper_is_open = open_gripper
            if open_gripper:
                self.gripper.goto(GRIPPER_MAX_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            else:
                self.gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE)

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


class Buttons(object):
    def __init__(self) -> None:
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def input(self) -> Tuple[bool, bool, bool]:
        # Get "A", "X", "Y" Button Presses
        pygame.event.get()
        a, x, y = self.gamepad.get_button(0), self.gamepad.get_button(2), self.gamepad.get_button(3)
        return a, x, y


class ArgumentParser(Tap):
    # fmt: off
    task: str                                           # Task ID for demonstration collection
    data_dir: Path = Path("data/demos/")                # Path to parent directory for saving demonstrations

    # Task Parameters
    include_visual_states: bool = False                 # Whether to run playback/get visual states (only False for now)
    max_time_per_demo: int = 15                         # Max time (in seconds) to record demo -- default = 21 seconds

    # Collection Parameters
    collection_strategy: str = "kinesthetic"            # How demos are collected :: only `kinesthetic` for now!
    controller: str = "joint"                           # Demonstration & playback uses a Joint Impedance controller...
    resume: bool = True                                 # Resume demonstration collection (on by default)
    # fmt: on


def demonstrate() -> None:
    args = ArgumentParser().parse_args()

    # Make directories for "raw" recorded states, and playback RGB states...
    #   > Note: the "record" + "playback" split is use for "kinesthetic" demos for obtaining visual state w/o humans!
    demo_raw_dir = args.data_dir / args.task / "record-raw"
    os.makedirs(demo_raw_dir, exist_ok=args.resume)
    if args.include_visual_states:
        demo_rgb_dir = args.data_dir / args.task / "playback-rgb"
        os.makedirs(demo_rgb_dir, exist_ok=args.resume)

    # Initialize environment in `record` mode...
    print("[*] Initializing Robot Connection...")
    env = FrankaEnv(
        home="default",
        hz=HZ,
        controller=args.controller,
        mode="record",
        use_camera=False,
        use_gripper=False,
    )

    # Initializing Button Control... TODO(siddk) -- switch with ASR
    print("[*] Connecting to Button Controller...")
    buttons, demo_index = Buttons(), 1

    # If `resume` -- get "latest" index
    if args.resume:
        files = os.listdir(demo_rgb_dir) if args.include_visual_states else os.listdir(demo_raw_dir)
        if len(files) > 0:
            demo_index = max([int(x.split("-")[-1].split(".")[0]) for x in files]) + 1

    # Start Recording Loop
    print("[*] Starting Demo Recording Loop...")
    while True:
        print(f"[*] Starting to Record Demonstration `{demo_index}`...")
        demo_file = f"{args.task}-{datetime.now().strftime('%m-%d')}-{demo_index}.npz"

        # Set `record`
        env.set_mode("record")

        # Reset environment & wait on user input...
        env.reset()
        print(
            "[*] Ready to record!\n"
            f"\tYou have `{args.max_time_per_demo}` secs to complete the demo, and can use (X) to stop recording.\n"
            "\tPress (Y) to reset, and (A) to start recording!\n "
        )

        # Loop on valid button input...
        a, _, y = buttons.input()
        while not a and not y:
            a, _, y = buttons.input()

        # Reset if (Y)...
        if y:
            continue

        # Go, go, go!
        print("\t=>> Started recording... press (X) to terminate recording!")

        # Drop into Recording Loop --> for `record` mode, we really only care about joint positions
        #   =>> TODO(siddk) - handle Gripper?
        joint_qs = []
        for _ in range(int(args.max_time_per_demo * HZ) - 1):
            # Get Button Input (only if True) --> handle extended button press...
            _, x, _ = buttons.input()

            # Terminate...
            if x:
                print("\tHit (X) - stopping recording...")
                break

            # Otherwise no termination, keep on recording...
            else:
                obs, _, _, _ = env.step(None)
                joint_qs.append(obs["q"])

        # Close Environment
        env.close()

        # Save "raw" demonstration...
        np.savez(str(demo_raw_dir / demo_file), hz=HZ, qs=joint_qs)

        # Enter Phase 2 -- Playback (Optional if not `args.include_visual_states`)
        do_playback = True
        if args.include_visual_states:
            print("[*] Entering Playback Mode - Please reset the environment to beginning and get out of the way!")
        else:
            # Loop on valid user button input...
            print("[*] Optional -- would you like to replay demonstration? Press (A) to playback, and (X) to continue!")
            a, x, _ = buttons.input()
            while not a and not x:
                a, x, _ = buttons.input()

            # Skip Playback!
            if x:
                do_playback = False

        # Special Playback Handling -- change gains, and replay!
        if do_playback:
            # TODO(siddk) -- handle Camera observation logging...
            env.set_mode("default")
            env.reset()

            # Block on User Ready -- Robot will move, so this is for safety...
            print("\tReady to playback! Get out of the way, and hit (A) to continue...")
            a, _, _ = buttons.input()
            while not a:
                a, _, _ = buttons.input()

            # Execute Trajectory
            print("\tReplaying...")
            for idx in range(len(joint_qs)):
                env.step(joint_qs[idx])

            # Close Environment
            env.close()

        # Move on?
        print("Next? Press (A) to continue or (Y) to quit... or (X) to retry demo and skip save")

        # Loop on valid button input...
        a, x, y = buttons.input()
        while not a and not y and not x:
            a, x, y = buttons.input()

        # Exit...
        if y:
            break

        # Bump Index
        if not x:
            demo_index += 1

    # And... that's all folks!
    print("[*] Done Demonstrating -- Cleaning Up! Thanks 🤖🚀")


if __name__ == "__main__":
    demonstrate()
