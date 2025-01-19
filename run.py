#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 03:10:55 2024

@author: aadi
"""

from lerobot.common.policies.act.modeling_act import ACTPolicy
import time
import torch
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
import cv2

inference_time_s = 60
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"
ckpt_path = "/home/revolabs/aditya/lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()
    # print("observation starting")
    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()
    # print("observation captured")
    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
            # Convert to a numpy array suitable for OpenCV
            # image = observation[name].cpu().numpy()  # Move to CPU and convert to numpy
            # image = (image * 255).astype("uint8")  # Convert to uint8 format
            # image = image.transpose(1, 2, 0)  # Convert back to HWC format for OpenCV
            # # Display the image using OpenCV
            # cv2.imshow("Video", image)
        
            # # Wait for a short period to simulate a video
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #   break  # Press 'q' to exit the video
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)
    # Compute the next action with the policy
    # based on the current observation
    # print("selecting action")
    action = policy.select_action(observation)
    # print("squeezing action")
    # Remove batch dimension
    action = action.squeeze(0)
    # print("action to cpu")
    # Move to cpu, if not already the case
    action = action.to("cpu")
    # print("action to robot")
    # Order the robot to move
    robot.send_action(action)
    dt_s = time.perf_counter() - start_time  
    busy_wait(1 / fps - dt_s)
    # print("end")
    
   
# follower_arm.write("Goal_Position", position)
current_pos = follower_arm.read("Present_Position")
steps = 30
for i in range(1, steps + 1):
    intermediate_pos = current_pos + (rest_position - current_pos) * (i / steps)
    follower_arm.write("Goal_Position", intermediate_pos)
    time.sleep(0.1) #try busy_wait