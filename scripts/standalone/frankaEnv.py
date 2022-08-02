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
from polymetis import GripperInterface, RobotInterface
from scipy.spatial.transform import Rotation as R


# === Polymetis Environment Wrapper ===
class FrankaEnv(Env):
    def __init__(
        self,
        home: str,
        hz: int,
        controller: str = "cartesian",
        mode: str = "default",
        step_size: float = 0.05,
        use_gripper: bool = False,
    ) -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param home: Default home position (specified as a string index into `HOMES` above!
        :param hz: Default policy control hz; somewhere between 20-40 is a good range.
        :param controller: Which impedance controller to use in < joint | cartesian | osc >
        :param mode: Mode in < "default" | ... > -- used to set P(D) gains!
        :param use_gripper: Boolean whether to initialize the Gripper controller (default: False)
        """
        self.home, self.rate, self.mode, self.controller, self.curr_step = home, Rate(hz), mode, controller, 0
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot, self.kp, self.kpd = None, None, None
        self.use_gripper, self.gripper, self.current_gripper_state, self.gripper_is_open = use_gripper, None, None, True

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