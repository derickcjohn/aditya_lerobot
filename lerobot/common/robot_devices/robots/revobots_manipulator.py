# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 03:43:07 2025

@author: aadi
"""
"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
# TODO(rcadene): reorganize the codebase into one file per robot, with the associated
# calibration procedure, to make it easy for people to add their own robot.

import json
import logging
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.robots.manipulator import ensure_safe_goal_position


@dataclass
class ManipulatorRobotConfig:
    """
    Example of usage:
    ```python
    ManipulatorRobotConfig()
    ```
    """
    # Define all components of the robot
    robot_type: str = "koch"
    leader_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})

    # Optionally limit the magnitude of the relative positional target vector for safety purposes.
    max_relative_target: list[float] | float | None = None

    # Optionally set the leader arm in torque mode with the gripper motor set to this angle.
    gripper_open_degree: float | None = None

    # === Extra Revobot configuration variables ===
    use_revobot_leader: bool = False
    use_revobot_follower: bool = False
    # New dictionaries to define Revobot arms (keyed by names such as "main")
    revobot_leader_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})
    revobot_follower_arms: dict[str, MotorsBus] = field(default_factory=lambda: {})

    def __setattr__(self, prop: str, val):
        if prop == "max_relative_target" and val is not None and isinstance(val, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(val):
                    raise ValueError(
                        f"len(max_relative_target)={len(val)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )
        super().__setattr__(prop, val)

    def __post_init__(self):
        if self.robot_type not in ["koch", "koch_bimanual", "aloha", "so100", "moss"]:
            raise ValueError(f"Provided robot type ({self.robot_type}) is not supported.")


class RevobotsManipulatorRobot:
    # TODO(rcadene): Implement force feedback
    """This class allows to control any manipulator robot of various number of motors.
    
    Non exhaustive list of robots:
    - Koch v1.0 / v1.1, Aloha, etc.
    
    Usage examples are provided in the class docstring.
    """

    def __init__(
        self,
        config: ManipulatorRobotConfig | None = None,
        calibration_dir: Path = ".cache/calibration/koch",
        **kwargs,
    ):
        if config is None:
            config = ManipulatorRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)
        self.calibration_dir = Path(calibration_dir)

        self.robot_type = self.config.robot_type
        
        # === Extra Revobot config initialization ===
        self.use_revobot_leader = self.config.use_revobot_leader
        self.use_revobot_follower = self.config.use_revobot_follower
        
        if self.use_revobot_leader:
            self.leader_arms = self.config.revobot_leader_arms
        else:
            self.leader_arms = self.config.leader_arms
            
        if self.use_revobot_follower:
            self.follower_arms = self.config.revobot_follower_arms
        else:
            self.follower_arms = self.config.follower_arms  
        
        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}


    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        # Only include arms if they are present.
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Allow connection if at least one device (arms or cameras) is specified.
        if not (self.leader_arms or self.follower_arms or self.cameras):
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in the docstring of the class."
            )

        # Connect the arms if they are present
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "moss"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # Disable torque on all motors before calibration
        for name in self.follower_arms:
            if self.use_revobot_follower == False:
                self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            if self.use_revobot_leader == False:
                self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        self.activate_calibration()

        # Set robot preset configurations based on robot_type
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["so100", "moss"]:
            self.set_so100_robot_preset()

        # Enable torque on follower arms if present
        for name in self.follower_arms:
            if self.use_revobot_follower == False:
                print(f"Activating torque on {name} follower arm.")
                self.follower_arms[name].write("Torque_Enable", 1)

        if self.config.gripper_open_degree is not None:
            if self.robot_type not in ["koch", "koch_bimanual"]:
                raise NotImplementedError(
                    f"{self.robot_type} does not support position AND current control in the handle, which is required to set the gripper open."
                )
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # Read initial positions from arms if they are present
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        
        NOTE: Revobot robots work without calibration, so if Revobot is in use, this step is skipped.
        """
        # if self.use_revobot_leader or self.use_revobot_follower:
        #     print("Revobot detected; skipping calibration.")
        #     return

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                print(f"Missing calibration file '{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration
                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)
                elif self.robot_type in ["so100", "moss"]:
                    from lerobot.common.robot_devices.robots.feetech_calibration import run_arm_manual_calibration
                    calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration
        
        if self.use_revobot_follower == False:
            for name, arm in self.follower_arms.items():
                calibration = load_or_run_calibration_(name, arm, "follower")
                arm.set_calibration(calibration)
        if self.use_revobot_leader == False:
            for name, arm in self.leader_arms.items():
                calibration = load_or_run_calibration_(name, arm, "leader")
                arm.set_calibration(calibration)

    def set_koch_robot_preset(self):
        def set_operating_mode_(arm):
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                arm.write("Operating_Mode", 4, all_motors_except_gripper)
                
            arm.write("Operating_Mode", 5, "gripper")
        
        if self.use_revobot_follower == False:
            for name in self.follower_arms:
                set_operating_mode_(self.follower_arms[name])
                self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
                self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
                self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        if self.use_revobot_leader == False:
            if self.config.gripper_open_degree is not None:
                for name in self.leader_arms:
                    set_operating_mode_(self.leader_arms[name])
                    self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                    self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")
    
    #TODO: compatible revobot with aloha
    def set_aloha_robot_preset(self):
        def set_shadow_(arm):
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")
            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        for name in self.follower_arms:
            set_shadow_(self.follower_arms[name])
        for name in self.leader_arms:
            set_shadow_(self.leader_arms[name])
        for name in self.follower_arms:
            self.follower_arms[name].write("Velocity_Limit", 131)
            all_motors_except_gripper = [name for name in self.follower_arms[name].motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                self.follower_arms[name].write("Operating_Mode", 4, all_motors_except_gripper)
            self.follower_arms[name].write("Operating_Mode", 5, "gripper")
        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree` is set to {self.config.gripper_open_degree}, but None is expected for Aloha instead",
                stacklevel=1,
            )

    def set_so100_robot_preset(self):
        if self.use_revobot_follower == False:
            for name in self.follower_arms:
                self.follower_arms[name].write("Mode", 0)
                self.follower_arms[name].write("P_Coefficient", 16)
                self.follower_arms[name].write("I_Coefficient", 0)
                self.follower_arms[name].write("D_Coefficient", 32)
                self.follower_arms[name].write("Lock", 0)
                self.follower_arms[name].write("Maximum_Acceleration", 254)
                self.follower_arms[name].write("Acceleration", 254)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # print("entering teleop")
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        leader_pos = {}
        if self.leader_arms:
            for name in self.leader_arms:
                before_lread_t = time.perf_counter()
                leader_pos[name] = self.leader_arms[name].read("Present_Position")
                leader_pos[name] = torch.from_numpy(leader_pos[name])
                self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        follower_goal_pos = {}
        if self.follower_arms:
            for name in self.follower_arms:
                # Only attempt to follow if corresponding leader data exists
                if name in leader_pos:
                    before_fwrite_t = time.perf_counter()
                    goal_pos = leader_pos[name]
                    if self.config.max_relative_target is not None:
                        present_pos = self.follower_arms[name].read("Present_Position")
                        present_pos = torch.from_numpy(present_pos)
                        goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)
                        # print(goal_pos)
                    follower_goal_pos[name] = goal_pos
                    goal_pos = goal_pos.numpy().astype(np.int32)
                    self.follower_arms[name].write("Goal_Position", values=goal_pos)
                    self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t
                else:
                    print(f"Skipping follower arm '{name}' since no corresponding leader data is available.")

        if not record_data:
            return

        follower_pos = {}
        if self.follower_arms:
            for name in self.follower_arms:
                before_fread_t = time.perf_counter()
                follower_pos[name] = self.follower_arms[name].read("Present_Position")
                follower_pos[name] = torch.from_numpy(np.array(follower_pos[name]))
                self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.cat([follower_pos[name] for name in self.follower_arms if name in follower_pos])
        action = torch.cat([follower_goal_pos[name] for name in self.follower_arms if name in follower_goal_pos])

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        follower_pos = {}
        if self.follower_arms:
            for name in self.follower_arms:
                before_fread_t = time.perf_counter()
                follower_pos[name] = self.follower_arms[name].read("Present_Position")
                follower_pos[name] = torch.from_numpy(follower_pos[name])
                self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.cat([follower_pos[name] for name in self.follower_arms if name in follower_pos])
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict = {"observation.state": state}
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        if not self.follower_arms:
            # No follower arms to command, so simply return the provided action.
            return action

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)
            action_sent.append(goal_pos)
            goal_pos = goal_pos.numpy().astype(np.int32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO: move robot-specific logs logic here

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        for name in self.follower_arms:
            if self.use_revobot_follower == False:
                self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            if self.use_revobot_leader == False:
                self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()
        for name in self.leader_arms:
            self.leader_arms[name].disconnect()
        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
