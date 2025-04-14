#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 00:27:19 2024

@author: aadi
"""

from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.policies.act.modeling_act import ACTPolicy
import time
from lerobot.scripts.control_robot import busy_wait
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import torch 
import cv2

leader_port = "/dev/ttyACM1"
follower_port = "/dev/ttyACM0"

leader_arm = DynamixelMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)
    
follower_arm = DynamixelMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
robot = ManipulatorRobot(
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",
    cameras={
        "phone": OpenCVCamera("/dev/video4", fps=30, width=640, height=480),
        "laptop": OpenCVCamera("/dev/video10", fps=30, width=640, height=480),
    },
)
robot.connect()
# rest_position = follower_arm.read("Present_Position")

# rest_position = leader_arm.read("Present_Position")
# rest_position = [  0.9667969 ,128.84766 ,  174.99023,   -16.611328,   -4.8339844  ,34.716797 ]
rest_position = [ -0.43945312, 117.509766, 118.916016, 85.78125, -4.482422, 34.716797  ]

inference_time_s = 10
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "/home/revolabs/aditya/aditya_lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

# say("I am going to collect objects")


for _ in range(inference_time_s * fps):
        start_time = time.perf_counter()
    
        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()
    
        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)
    
        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        # Remove batch dimension
        action = action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        # Order the robot to move
        robot.send_action(action)
    
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)

cv2.destroyAllWindows()

current_pos = follower_arm.read("Present_Position")
steps = 30
for i in range(1, steps + 1):
        intermediate_pos = current_pos + (rest_position - current_pos) * (i / steps)
        follower_arm.write("Goal_Position", intermediate_pos)
        time.sleep(0.1) #try busy_wait
   

robot.disconnect()

#ffplay -f video4linux2 -framerate 10 -video_size 480x270 /dev/video10
#ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental -rtsp_transport tcp rtsp://admin:robot123@192.168.0.113:8554/Streaming/Channels/101
#ffplay -probesize 32 -analyzeduration 0 -sync ext -fflags nobuffer -flags low_delay -framedrop -strict experimental -rtsp_transport tcp rtsp://admin:robot123@192.168.0.113:8554/Streaming/Channels/102

# [  0.9667969 ,128.84766 ,  174.99023,   -16.611328,   -4.8339844  ,34.716797 ]

# [ -2.9882812  136.05469    179.29688     1.7578125    -7.1191406  34.804688 ]