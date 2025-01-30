# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 06:55:10 2025

@author: aadi
"""

from lerobot.common.policies.act.modeling_act import ACTPolicy
import time
from lerobot.scripts.control_robot import busy_wait
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import torch 
import os
import platform
import cv2

record_time_s = 30
fps = 60

states = []
actions = []

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

robot = ManipulatorRobot(
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",
    cameras={
        "phone": OpenCVCamera("/dev/video6", fps=30, width=640, height=480),
        "laptop": OpenCVCamera("/dev/video4", fps=30, width=640, height=480),
    },
)

robot.connect()

rest_position = follower_arm.read("Present_Position")


def say(text, blocking=False):
        # Check if mac, linux, or windows.
        if platform.system() == "Darwin":
            cmd = f'say "{text}"'
        elif platform.system() == "Linux":
            cmd = f'spd-say "{text}"'
        elif platform.system() == "Windows":
            cmd = (
                'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')\""
            )
    
        if not blocking and platform.system() in ["Darwin", "Linux"]:
            # TODO(rcadene): Make it work for Windows
            # Use the ampersand to run command in the background
            cmd += " &"
    
        os.system(cmd)

def capture_all_coordinates(robot):
        """
        Captures a single coordinate (x, y) for each image in the robot observation 
        by asking the user to double-click on the displayed images.
        
        Returns:
        dict: A dictionary of {image_key: (x, y)} or None if the user pressed Esc.
        """
        coords_dict = {}
        
        # 1) Capture the current state/observation from the robot
        observation = robot.capture_observation()
        
        # 2) Find all keys that contain 'image'
        image_keys = [key for key in observation if "image" in key]
        
        # 3) For each image key, capture a coordinate
        for key in image_keys:
            # Convert from torch Tensor (RGB) to NumPy (BGR) if needed
            # If observation[key] is already a NumPy array in RGB:
            img_rgb = observation[key].numpy()  # shape (H, W, 3) or (3, H, W)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
            print(f"Please double-click on the window to select a point for '{key}'")
            coords = cursor_coordinate_input(img_bgr, window_name=str(key))
        
            # If user pressed Esc without double-click, coords will be None
            if coords is None:
                print(f"No coordinates captured for '{key}' (user pressed Esc).")
                coords_dict[key] = None
            else:
                print(f"Coordinates for '{key}': {coords}")
                coords_dict[key] = coords
        
        # 4) Print and return the entire dictionary of coordinates
        print("All coordinates:", coords_dict)
        return coords_dict


def cursor_coordinate_input(image_path, window_name=None):
        """
        Displays the image at image_path, waits for a double-click from the user,
        and returns the (x, y) coordinates of the double-click point in the image.
        
        Returns:
        (x, y) coordinates as a tuple of integers if a double-click happens;
        None if the image cannot be loaded or if the user exits without a double-click.
        """
        
        # Try to load the image
        if image_path is not str:
            img = image_path
        else:
            img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot load image from {image_path}")
            return None
        
        # A dictionary to hold the coordinate once a double-click event is detected
        coord_container = {'coords': None}
        
        # Define the mouse callback
        def on_mouse(event, x, y, flags, param):
            # Check if the left mouse button was double clicked
            if event == cv2.EVENT_LBUTTONDBLCLK:
                param['coords'] = (x, y)
        
        # Create a named window and set the mouse callback
        window_name = window_name+ " " +"Double-Click to Capture Coordinates"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse, coord_container)
        
        # Display the image and wait for a double-click or Esc key
        while True:
            cv2.imshow(window_name, img)
            key = cv2.waitKey(1) & 0xFF
        
            # If the user has double-clicked, break out of the loop
            if coord_container['coords'] is not None:
                break
        
            # If 'Esc' is pressed, break out without returning coordinates
            if key == 27:  # 27 is the Esc key
                break
        
        # Clean up windows
        cv2.destroyAllWindows()
        
        return coord_container['coords']


def put_the_marker(
    image,
    name,
    coords_dict,
    radius=10,
    border_color=(0, 0, 255),
    cross_color=(0, 0, 255),
    bg_color=(255, 255, 255)
):
        """
        Draws a circle with a white background, red border, and a red cross in the center
        at the coordinates corresponding to the provided `name` in `coords_dict`.
        
        Args:
        image: The frame/image (NumPy array) to draw on.
        name: The key (string) to look up in coords_dict, e.g. 'observation.images.phone'.
        coords_dict: A dictionary mapping keys (e.g. 'observation.images.phone') to (x, y) coordinates.
        radius: Radius of the circle.
        border_color: BGR tuple for the circle's border color (default: red).
        cross_color: BGR tuple for the cross color (default: red).
        bg_color: BGR tuple for the circle's background color (default: white).
        
        The function does nothing if `name` is not found in `coords_dict` or if its value is None.
        """
        
        # Check if the name exists in the dictionary and has valid coordinates
        if name not in coords_dict:
            print(f"put_the_marker: No key named '{name}' in coords_dict.")
            return
        
        center = coords_dict[name]
        if center is None:
            print(f"put_the_marker: '{name}' has no coordinates (None).")
            return
        
        # Now we have a valid (x, y)
        x, y = center
        
        # 1) Draw the filled circle for background
        cv2.circle(image, center, radius, bg_color, -1)
        
        # 2) Draw the border
        cv2.circle(image, center, radius, border_color, 2)
        
        # 3) Draw the cross inside the circle
        #    - Vertical line
        cv2.line(image, (x, y - (radius - 1)), (x, y + (radius - 1)), cross_color, 2)
        #    - Horizontal line
        cv2.line(image, (x - (radius - 1), y), (x + (radius - 1), y), cross_color, 2)
        
        return image
    

coords_dict = capture_all_coordinates(robot)


inference_time_s = 120
fps = 30
device = "cuda"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "/home/revolabs/aditya/aditya_lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

say("I am going to collect objects")


for _ in range(inference_time_s * fps):
        start_time = time.perf_counter()
        
        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()
        
        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                
                if coords_dict is not None:
                    img_bgr = cv2.cvtColor(observation[name].numpy(), cv2.COLOR_RGB2BGR)
                    # OpenCV expects (x, y) as (center_x, center_y)
                    put_the_marker(img_bgr,name,coords_dict)
                    img_rgb = cv2.cvtColor(observation[name].numpy(), cv2.COLOR_BGR2RGB)
                    if "laptop" in name:
                        cv2.imshow("laptop", img_bgr)
                        key = cv2.waitKey(1)
                    
                    if "phone" in name:
                        cv2.imshow("phone", img_bgr)
                        key = cv2.waitKey(1)
                    
                    observation[name] = torch.from_numpy(img_rgb)
            
                    
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
say("I have finished collecting objects")

current_pos = follower_arm.read("Present_Position")
steps = 30
for i in range(1, steps + 1):
        intermediate_pos = current_pos + (rest_position - current_pos) * (i / steps)
        follower_arm.write("Goal_Position", intermediate_pos)
        time.sleep(0.1) #try busy_wait
   

robot.disconnect()