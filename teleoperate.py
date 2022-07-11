"""
teleoperate.py

Run 6-DoF teleoperation of the Franka Panda arm, using an attached Joystick with 2 joystick axes and a single,
trigger-based modal control. Supports various action scales and gripper control (via a toggle).
"""
from odyssey.robot import FrankaEnv


def teleoperate() -> None:
    print("[*] Starting Teleoperation!")

    # Connect to the Robot
    _ = FrankaEnv(home="default")


if __name__ == "__main__":
    teleoperate()
