#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:04:02 2025

@author: aadi
"""

import os
import sys
import cv2
import time
import torch
import platform
import threading
from PIL import Image
from typing import List
import numpy as np

# For generative AI bounding-box detection (if needed)
import google.generativeai as genai

from pynput.keyboard import Key, Listener

# Robot-specific imports
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
from lerobot.common.robot_devices.cameras.intelrealsense import IntelRealSenseCamera
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.scripts.control_robot import busy_wait
from lerobot.common.policies.act.modeling_act import ACTPolicy


################################################################################
# Global variables
################################################################################

program_ending = False
manual_detection = True  # If True -> double-click in "phone" window for coords

# In manual_detection mode, we store the user's double-click here
clicked_coords = None  # Will hold (x, y) or None

# Store latest frames for cameras
img_buffer = {
    "phone": None,
    "laptop": None
}

################################################################################
# Utility Functions
################################################################################

def say(text, blocking=False):
    """
    Simple TTS wrapper that works on macOS, Linux, or Windows.
    """
    if platform.system() == "Darwin":
        cmd = f'say "{text}"'
    elif platform.system() == "Linux":
        cmd = f'spd-say "{text}"'
    elif platform.system() == "Windows":
        cmd = (
            'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
            f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
        )
    else:
        print(f"(say) OS not recognized. Printing text: {text}")
        return

    if not blocking and platform.system() in ["Darwin", "Linux"]:
        # On Windows we skip running in background for simplicity
        cmd += " &"
    os.system(cmd)


def put_the_marker(
    image: np.ndarray,
    coords: tuple,
    radius=10,
    border_color=(0, 0, 255),  # red
    cross_color=(0, 0, 255),   # red
    bg_color=(255, 255, 255),  # white fill
):
    """
    Draw a marker on the given image at the specified `coords`.
    """
    if coords is None:
        return image  # no marker

    x, y = coords
    # Circle background
    cv2.circle(image, (x, y), radius, bg_color, -1)
    # Circle border
    cv2.circle(image, (x, y), radius, border_color, 2)
    # Cross inside
    cv2.line(image, (x, y - (radius - 1)), (x, y + (radius - 1)), cross_color, 2)
    cv2.line(image, (x - (radius - 1), y), (x + (radius - 1), y), cross_color, 2)
    return image


def on_mouse_double_click(event, x, y, flags, param):
    """
    Mouse callback that captures the (x, y) on double-click.
    Stores result in global `clicked_coords`.
    """
    global clicked_coords
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked_coords = (x, y)
        print(f"[Manual Detection] Double-click at: {clicked_coords}")


def object_detection(image: np.ndarray, object_to_detect: str) -> List[tuple]:
    """
    Example function that uses PaLM API (via google.generativeai)
    to detect bounding boxes or centroids for 'object_to_detect' in `image`.
    Returns a list of (centroid_x, centroid_y).

    This is a STUB—adjust the bounding box logic to your needs.
    """
    # Configure your API key
    GOOGLE_API_KEY = "AIzaSyBzC0kdp5WKkhaHQVWvqpFdvoCGCvdyCIE"
    genai.configure(api_key=GOOGLE_API_KEY)

    # Convert OpenCV (BGR) image to PIL (RGB)
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_rgb)

    height, width, _ = image.shape
    
    # Example prompt. Adjust to your actual bounding-box logic.
    prompt = f"""Please detect {object_to_detect} in this image. For each object, output:
         
        1. The object label created (e.g., "phone", "cable").
        2. Its bounding box in [y_min, x_min, y_max, x_max] format.
    
        Use the exact format:
          label:[y_min, x_min, y_max, x_max]
    
        One object per line, with no additional heading, text or explanation."""

    model = genai.GenerativeModel(model_name="learnlm-1.5-pro-experimental")
    response = model.generate_content([pil_img, prompt])

    coords_list = []
    lines = response.text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if (':' not in line) or not line:
            continue

        label_part, bbox_part = line.split(':', 1)
        bbox_part = bbox_part.strip()

        if not (bbox_part.startswith('[') and bbox_part.endswith(']')):
            continue

        coords_str = bbox_part[1:-1].strip()
        parts = coords_str.split(',')
        if len(parts) != 4:
            continue

        y_min, x_min, y_max, x_max = [int(p.strip()) for p in parts]

        # Here we're assuming the model might have returned normalized coords
        # or something in the range 1000, etc. Adjust as needed.
        y_min_pix = int((y_min / 1000.0) * height)
        x_min_pix = int((x_min / 1000.0) * width)
        y_max_pix = int((y_max / 1000.0) * height)
        x_max_pix = int((x_max / 1000.0) * width)

        centroid_x = (x_min_pix + x_max_pix) // 2
        centroid_y = (y_min_pix + y_max_pix) // 2

        coords_list.append((centroid_x, centroid_y))
    
    coords_list.sort(key=lambda coord: coord[0])

    return coords_list


################################################################################
# Thread: Camera Loop
################################################################################

def camera_thread_func(robot):
    """
    Continuously read frames from the robot's "phone" camera, update `img_buffer`, 
    and display the frame in an OpenCV window.

    If `manual_detection` is True, set a mouse callback on the "phone" window
    to capture double-click coordinates. Otherwise, just display frames 
    with no callback.
    """
    global program_ending, clicked_coords, manual_detection

    # Named window
    cv2.namedWindow("phone", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("phone", 640, 480)

    if manual_detection:
        # Set mouse callback on the phone camera window
        cv2.setMouseCallback("phone", on_mouse_double_click)

    while not program_ending:
        # Grab frame from the phone camera
        frame = robot.cameras["phone"].async_read()
        if frame is not None:
            img_buffer["phone"] = frame.copy()

            # Convert to BGR for OpenCV display
            display_frame = cv2.cvtColor(img_buffer["phone"], cv2.COLOR_RGB2BGR)

            # If manual detection, put the marker wherever the user double-clicked
            if clicked_coords is not None:
                put_the_marker(display_frame, clicked_coords)

            cv2.imshow("phone", display_frame)

        # Check for 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            program_ending = True
            break

    # Cleanup
    cv2.destroyAllWindows()
    print("[Camera Thread] Exiting camera loop.")


################################################################################
# Detection Function
################################################################################

def detect_target_coords():
    """
    Detects the target coordinate either manually (by double-click) or 
    via AI (object_detection function). Updates the global clicked_coords 
    if applicable, and returns the final chosen coordinate (or None).
    """
    global clicked_coords, manual_detection, img_buffer

    coords_list = []

    # If user is manually detecting via double-click
    if manual_detection:
        say("Please double-click on the phone window to mark your object. Press ENTER once done.")
        print("[Manual Detection] Press ENTER in terminal once done selecting points.")

        def on_release_enter(key):
            if key == Key.enter:
                return False

        # Wait until user presses ENTER
        with Listener(on_release=on_release_enter) as listener:
            listener.join()

        # Now we have clicked_coords if the user double-clicked
        if clicked_coords is not None:
            coords_list = [clicked_coords]
        else:
            print("[Manual Detection] No point was clicked.")

    # Otherwise, use AI-based detection
    else:
        object_to_detect = input("Enter object prompt to detect: ")
        frame_phone = img_buffer["phone"]
        if frame_phone is not None:
            frame_bgr = cv2.cvtColor(frame_phone, cv2.COLOR_RGB2BGR)
            coords_list = object_detection(frame_bgr, object_to_detect)
            print("[AI Detection] coords_list:", coords_list)
        else:
            print("[AI Detection] No phone frame in buffer!")

    # If multiple coords, let user choose
    if len(coords_list) > 1:
        print("Multiple objects found. Indices from left to right:")
        for idx, c in enumerate(coords_list):
            print(f"  Index={idx}, coords={c}")
        chosen_idx = int(input("Choose index to track: "))
        if 0 <= chosen_idx < len(coords_list):
            coords_list = [coords_list[chosen_idx]]
        else:
            print("[AI Detection] Invalid index. Defaulting to the first.")
            coords_list = [coords_list[0]]

    # Either set clicked_coords to the chosen detection or None
    if coords_list:
        clicked_coords = coords_list[0]
    else:
        clicked_coords = None

    return clicked_coords


################################################################################
# Helper: Go to Home Position
################################################################################

def go_to_home_position(robot, rest_position):
    """
    Moves the follower_arm from its current position to `rest_position`
    in `steps` small increments.
    """
    current_pos = robot.follower_arms['main'].read("Present_Position")
    steps=30
    for i in range(1, steps + 1):
        alpha = i / steps
        intermediate_pos = current_pos + alpha * (rest_position - current_pos)
        robot.follower_arms['main'].write("Goal_Position", intermediate_pos)
        time.sleep(0.05)


################################################################################
# Inference Function
################################################################################

def run_inference(robot, rest_position):
    """
    Perform the entire model-loading and inference loop, then
    return the arm to the rest position. This includes drawing
    the user-selected marker onto the phone image during inference.
    """
    global clicked_coords

    inference_time_s = 10
    fps = 30
    device = "cuda"  # or "cpu"

    ckpt_path = "/home/revolabs/cs_capstone/lerobot/outputs/train/act_koch_reach_the_marker/pretrained_model"
    # ckpt_path = "/home/revolabs/cs_capstone/lerobot/outputs/train/act_koch_follow_marker_2/last/pretrained_model"

    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.to(device)

    print(f"Running inference for {inference_time_s} seconds...")
    for _ in range(int(inference_time_s * fps)):
        t0 = time.perf_counter()

        # Get observation
        observation = robot.capture_observation()

        # Convert images to the correct shape/range, add marker if needed
        for name in observation:
            if "image" in name:
                if clicked_coords is not None and "phone" in name:
                    # Convert to BGR, draw marker, convert back
                    img_bgr = cv2.cvtColor(observation[name].numpy(), cv2.COLOR_RGB2BGR)
                    put_the_marker(img_bgr, clicked_coords)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    observation[name] = torch.from_numpy(img_rgb)

                # Convert from [H,W,C] in [0..255] to float32 in [0..1], permute dims
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)
        
        # Compute the next action with the policy
        action = policy.select_action(observation)
        action = action.squeeze(0).to("cpu")
        robot.send_action(action)

        # Keep constant fps
        dt = time.perf_counter() - t0
        busy_wait(max(0, 1.0 / fps - dt))

    print("[Inference] Complete. Returning to rest position.")
    # Use the helper function to move home
    go_to_home_position(robot, rest_position)


################################################################################
# Main
################################################################################

def main():
    global program_ending, clicked_coords, manual_detection

    say("Starting up...", blocking=False)

    #######################################################
    # Setup Robot
    #######################################################
    
    leader_port = "/dev/ttyACM1"
    follower_port = "/dev/ttyACM0"

    leader_arm = DynamixelMotorsBus(
        port=leader_port,
        motors={
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
            # "phone": OpenCVCamera("/dev/video12", fps=30, width=640, height=480),
            # "laptop": OpenCVCamera("/dev/video10", fps=30, width=640, height=480),
            "phone": IntelRealSenseCamera("828612060404", fps=30, width=640, height=480),
            "laptop": IntelRealSenseCamera("816612060176", fps=30, width=640, height=480),
        },
    )

    robot.connect()

    # Read the rest position here so we can pass it to the inference function
    # rest_position = follower_arm.read("Present_Position")
    rest_position = [  0.9667969 ,128.84766 ,  174.99023,   -16.611328,   -4.8339844  ,34.716797 ]

    #######################################################
    # Start Camera Thread
    #######################################################
    cam_thread = threading.Thread(target=camera_thread_func, args=(robot,))
    cam_thread.start()

    # Small delay to allow initial frames
    time.sleep(2.0)

    #######################################################
    # DETECTION (Manual or AI) as a function
    #######################################################
    detect_target_coords()  # sets clicked_coords internally

    #######################################################
    # RUN INFERENCE
    #######################################################
    run_inference(robot, rest_position)

    # Cleanup
    say("Done.")
    program_ending = True
    cam_thread.join()
    robot.disconnect()
    print("[Main] Program ended gracefully.")


if __name__ == "__main__":
    main()
