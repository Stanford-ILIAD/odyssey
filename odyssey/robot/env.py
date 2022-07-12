"""
env.py

Core abstraction over the physical Robot hardware, sensors, and internal robot state. Follows a standard OpenAI Gym API.
"""
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from gym import Env
from polymetis import RobotInterface

from .util import HOMES, HZ, Rate, quat2euler


class FrankaEnv(Env):
    def __init__(self, home: str, hz: int = HZ, controller: str = "osc") -> None:
        """
        Initialize a *physical* Franka Environment, with the given home pose, PD controller gains, and camera.

        :param home: Default home position (specified as a string index into `HOMES` above!)
        :param hz: Default policy control Hz; somewhere between 20-60 is a good range.
        :param controller: Which controller to use in < joint | cartesian | osc > (teleoperation uses `osc`)
        """
        self.home, self.rate, self.controller, self.curr_step = home, Rate(hz), controller, 0
        self.current_joint_pose, self.current_ee_pose, self.current_ee_rot = None, None, None
        self.robot = None

        # Initialize Robot and PD Controller
        self.reset()

    def robot_setup(self, franka_ip: str = "172.16.0.1") -> None:
        # Initialize Robot Interface and Reset to Home
        self.robot = RobotInterface(ip_address=franka_ip)
        self.robot.set_home_pose(torch.Tensor(HOMES.get(self.home, HOMES["default"])))
        self.robot.go_home()

        # Initialize current joint & EE poses...
        self.current_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        self.current_ee_rot = quat2euler(self.current_ee_pose[3:])
        self.current_joint_pose = self.robot.get_joint_positions().numpy()

        # Create an *Impedance Controller*, with the desired gains...
        #   > Note: Feel free to add any other controller, e.g., a PD controller around joint poses.
        #           =>> Ref: https://github.com/AGI-Labs/franka_control/blob/master/util.py#L74
        if self.controller == "joint":
            self.robot.start_joint_impedance()
        elif self.controller in ["cartesian", "osc"]:
            self.robot.start_cartesian_impedance()
        else:
            raise NotImplementedError(f"Support for controller `{self.controller}` not yet implemented!")

    def reset(self) -> Dict[str, np.ndarray]:
        self.robot_setup()
        return self.get_obs()

    def get_obs(self) -> Dict[str, np.ndarray]:
        new_ee_pose = np.concatenate([a.numpy() for a in self.robot.get_ee_pose()])
        new_ee_rot = quat2euler(new_ee_pose[3:])
        new_joint_pose = self.robot.get_joint_positions().numpy()

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
                self.robot.update_desired_joint_positions(torch.from_numpy(action))
            elif self.controller == "cartesian":
                # First 3 elements are xyz, last 4 elements are quaternion orientation...
                self.robot.update_desired_ee_pose(
                    position=torch.from_numpy(action[:3]), orientation=torch.from_numpy(action[3:])
                )
            elif self.controller == "osc":
                # First 3 elements are xyz, last 3 elements are euler angles...
                print("Oh no, OSC action not yet defined!")
                import IPython

                IPython.embed()
            else:
                raise NotImplementedError(f"Controller type `{self.controller}` is not yet implemented!")

        # Sleep according to control frequency
        self.rate.sleep()

        # Return observation, Gym default signature...
        return self.get_obs(), 0, False, None

    @property
    def ee_position(self) -> np.ndarray:
        """Return current EE position --> 3D (x, y, z)!"""
        return self.current_ee_pose[:3]

    @property
    def ee_orientation(self) -> np.ndarray:
        """Return current EE orientation as euler angles (in radians) --> 3D roll/pitch/yaw!"""
        return self.current_ee_rot

    def render(self, mode: str = "human") -> None:
        raise NotImplementedError("Render is not implemented for Physical FrankaEnv...")

    def close(self) -> Any:
        # Terminate Policy
        logs = self.robot.terminate_current_policy()

        # Garbage collection & sleep just in case...
        del self.robot
        self.robot = None, None
        time.sleep(1)

        return logs
