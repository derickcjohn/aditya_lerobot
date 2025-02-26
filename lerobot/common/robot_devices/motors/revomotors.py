#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:20:28 2025

@author: revolabs
"""

import enum
import logging
import math
import time
import traceback
from copy import deepcopy

import numpy as np
import tqdm
import socket

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

class RevobotRobotBus:
    def __init__(self, socket_ip: str, socket_port: int, motors: dict[str, tuple[int, str]]):
        """
        Initialize the Revobot robot bus with the given socket details and motors dictionary.
        :param socket_ip: The IP address for the Revobot connection.
        :param socket_port: The port number for the Revobot connection.
        :param motors: A dictionary mapping motor names to a tuple (motor_id, model).
                       Expected to have 6 motors (same as Koch).
        """
        self.socket_ip = socket_ip
        self.socket_port = socket_port
        self.motors = motors
        self.sock = None
        self.calibration = None
        print(f"RevobotRobotBus.__init__ called with socket {self.socket_ip}:{self.socket_port}")

    def connect(self):
        print("RevobotRobotBus.connect called")
        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # try:
        #     self.sock.connect((self.socket_ip, self.socket_port))
        #     print(f"Connected to Revobot at {self.socket_ip}:{self.socket_port}")
        # except Exception as e:
        #     print("Failed to connect to Revobot:", e)
        #     self.sock = None

    def reconnect(self):
        print("RevobotRobotBus.reconnect called")
        if self.sock:
            self.disconnect()
        self.connect()

    def are_motors_configured(self):
        print("RevobotRobotBus.are_motors_configured called")
        # For simulation, assume they are configured.
        return True

    def find_motor_indices(self):
        print("RevobotRobotBus.find_motor_indices called")
        return list(self.motors.keys())

    def set_bus_baudrate(self, baudrate):
        print("RevobotRobotBus.set_bus_baudrate called with baudrate", baudrate)
        # Not applicable for socket communication.

    @property
    def motor_names(self):
        print("RevobotRobotBus.motor_names property accessed")
        return list(self.motors.keys())

    @property
    def motor_models(self):
        print("RevobotRobotBus.motor_models property accessed")
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self):
        print("RevobotRobotBus.motor_indices property accessed")
        return [idx for idx, _ in self.motors.values()]

    def set_calibration(self, calibration: dict):
        print("RevobotRobotBus.set_calibration called with calibration:", calibration)
        self.calibration = calibration

    def apply_calibration_autocorrect(self, values, motor_names=None):
        print("RevobotRobotBus.apply_calibration_autocorrect called")
        return values

    def apply_calibration(self, values, motor_names=None):
        print("RevobotRobotBus.apply_calibration called")
        return values

    def autocorrect_calibration(self, values, motor_names=None):
        print("RevobotRobotBus.autocorrect_calibration called")
        return values

    def revert_calibration(self, values, motor_names=None):
        print("RevobotRobotBus.revert_calibration called")
        return values

    def read_with_motor_ids(self, motor_ids, data_name, num_retry=10):
        print("RevobotRobotBus.read_with_motor_ids called for data_name:", data_name)
        # Simulation: Return a list of zeros.
        return [0] * len(motor_ids)

    def read(self, data_name, motor_names=None):
        print("RevobotRobotBus.read called for", data_name)
        # Simulation: Return a numpy array of zeros with length equal to number of motors.
        return np.zeros(len(self.motors), dtype=np.int32)

    def write_with_motor_ids(self, motor_ids, data_name, values, num_retry=10):
        print("RevobotRobotBus.write_with_motor_ids called for", data_name, "with values:", values)
        # Simulation: Do nothing.

    def write(self, data_name, values, motor_names=None):
        print("RevobotRobotBus.write called for", data_name, "with values:", values)
        # Simulation: Attempt to send data over the socket.
        # if self.sock:
        #     try:
        #         # Convert values to string and encode.
        #         self.sock.sendall(str(values).encode('utf-8'))
        #     except Exception as e:
        #         print("Socket send failed:", e)
        # else:
        #     print("Socket not connected; cannot send data.")

    def disconnect(self):
        print("RevobotRobotBus.disconnect called")
        # if self.sock:
        #     self.sock.close()
        #     self.sock = None

    def __del__(self):
        print("RevobotRobotBus.__del__ called")
        self.disconnect()
