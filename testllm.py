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
import speech_recognition as sr
import base64
from dotenv import load_dotenv, find_dotenv
import argparse

# For generative AI bounding-box detection (if needed)
import google.generativeai as genai

from pynput.keyboard import Key, Listener

# Robot-specific imports
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

# Policy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect

# LangChain / ChatGPT Agent imports (as requested)
from langchain.schema import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

################################################################################
# Global variables
################################################################################

program_ending = False
clicked_coords = None    # Will hold (x, y) or None
use_manual_detection = None  # Global flag: True if user chooses manual detection

# Store latest frames for cameras
img_buffer = {"phone": None, "laptop": None}

# Globals to be set in main()
robot = None
rest_position = None
policies = {}  # Dictionary to hold policy configurations

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
        cmd += " &"
    os.system(cmd)

def put_the_marker(image: np.ndarray, coords: tuple, radius=10,
                   border_color=(0, 0, 255), cross_color=(0, 0, 255),
                   bg_color=(255, 255, 255)):
    """
    Draw a marker on the given image at the specified `coords`.
    """
    if coords is None:
        return image

    x, y = coords
    cv2.circle(image, (x, y), radius, bg_color, -1)
    cv2.circle(image, (x, y), radius, border_color, 2)
    cv2.line(image, (x, y - (radius - 1)), (x, y + (radius - 1)), cross_color, 2)
    cv2.line(image, (x - (radius - 1), y), (x + (radius - 1), y), cross_color, 2)
    return image

def on_mouse_double_click(event, x, y, flags, param):
    """
    Mouse callback that captures the (x, y) on double-click.
    Only updates clicked_coords if manual detection is active.
    """
    global clicked_coords, use_manual_detection
    if use_manual_detection and event == cv2.EVENT_LBUTTONDBLCLK:
        clicked_coords = (x, y)
        print(f"[Manual Detection] Double-click at: {clicked_coords}")

def object_detection(image: np.ndarray, object_to_detect: str) -> List[tuple]:
    """
    Example function that uses PaLM API (via google.generativeai)
    to detect bounding boxes or centroids for 'object_to_detect' in `image`.
    Returns a list of (centroid_x, centroid_y).

    This is a STUB—adjust the bounding box logic to your needs.
    """
    GOOGLE_API_KEY = "AIzaSyBzC0kdp5WKkhaHQVWvqpFdvoCGCvdyCIE"
    genai.configure(api_key=GOOGLE_API_KEY)

    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_rgb)
    height, width, _ = image.shape
    prompt = f"""Please detect {object_to_detect} in this image. For each object, output:
         
        1. The object label (e.g., "phone", "cable").
        2. Its bounding box in [y_min, x_min, y_max, x_max] format.
    
        Use the exact format:
          label:[y_min, x_min, y_max, x_max]
    
        One object per line, with no additional text."""
    
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
        y_min_pix = int((y_min / 1000.0) * height)
        x_min_pix = int((x_min / 1000.0) * width)
        y_max_pix = int((y_max / 1000.0) * height)
        x_max_pix = int((x_max / 1000.0) * width)
        centroid_x = (x_min_pix + x_max_pix) // 2
        centroid_y = (y_min_pix + y_max_pix) // 2
        coords_list.append((centroid_x, centroid_y))
    coords_list.sort(key=lambda coord: coord[0])
    return coords_list

def camera_thread_func(robot):
    """
    Continuously read frames from the robot's "phone" camera, update `img_buffer`, 
    and display the frame in an OpenCV window.
    """
    global program_ending, clicked_coords, use_manual_detection
    cv2.namedWindow("phone", cv2.WINDOW_NORMAL)
    # Always set the callback; it only updates if use_manual_detection is True.
    cv2.setMouseCallback("phone", on_mouse_double_click)
    while not program_ending:
        frame = robot.cameras["phone"].async_read()
        if frame is not None:
            img_buffer["phone"] = frame.copy()
            display_frame = cv2.cvtColor(img_buffer["phone"], cv2.COLOR_RGB2BGR)
            if clicked_coords is not None:
                put_the_marker(display_frame, clicked_coords)
            cv2.imshow("phone", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            program_ending = True
            break
    cv2.destroyAllWindows()
    print("[Camera Thread] Exiting camera loop.")

def detect_target_coords():
    """
    Detects the target coordinate either manually (by double-click) or via AI.
    When called, it asks the user if they want to use manual detection.
    Updates the global clicked_coords and returns it.
    """
    global clicked_coords, img_buffer, use_manual_detection
    answer = input("Use manual detection for target? (y/n): ").strip().lower()
    use_manual_detection = (answer == "y")
    coords_list = []
    if use_manual_detection:
        say("Please double-click on the phone window to mark your object. Then press ENTER in terminal when done.")
        input("Press ENTER when done selecting points...")
        if clicked_coords is not None:
            coords_list = [clicked_coords]
        else:
            print("[Manual Detection] No point was clicked.")
    else:
        object_to_detect = input("Enter object prompt to detect: ")
        frame_phone = img_buffer["phone"]
        if frame_phone is not None:
            frame_bgr = cv2.cvtColor(frame_phone, cv2.COLOR_RGB2BGR)
            coords_list = object_detection(frame_bgr, object_to_detect)
            print("[AI Detection] coords_list:", coords_list)
        else:
            print("[AI Detection] No phone frame in buffer!")
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
    if coords_list:
        clicked_coords = coords_list[0]
    else:
        clicked_coords = None
    return clicked_coords

def go_to_home_position(robot, rest_position):
    """
    Moves the follower arm from its current position to `rest_position`
    in small increments.
    """
    current_pos = robot.follower_arms['main'].read("Present_Position")
    steps = 30
    for i in range(1, steps + 1):
        alpha = i / steps
        intermediate_pos = current_pos + alpha * (rest_position - current_pos)
        robot.follower_arms['main'].write("Goal_Position", intermediate_pos)
        time.sleep(0.05)

def run_inference(robot, rest_position, chkpt, control_time_s):
    """
    Loads the policy from the given checkpoint and runs inference (acting like a control loop)
    for `control_time_s` seconds. At each step, the robot captures an observation, draws a marker
    if one was selected, computes an action using the policy, and sends the action to the robot.
    This function no longer returns the robot to the rest position automatically.
    """
    global use_manual_detection
    fps = 30
    device = "cuda"  # or "cpu"
    policy = ACTPolicy.from_pretrained(chkpt)
    policy.to(device)
    print(f"Running inference for {control_time_s} seconds using checkpoint {chkpt}...")
    for _ in range(int(control_time_s * fps)):
        t0 = time.perf_counter()
        observation = robot.capture_observation()
        for name in observation:
            if "image" in name:
                if clicked_coords is not None and "phone" in name:
                    img_bgr = cv2.cvtColor(observation[name].numpy(), cv2.COLOR_RGB2BGR)
                    put_the_marker(img_bgr, clicked_coords)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    observation[name] = torch.from_numpy(img_rgb)
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0).to(device)
        action = policy.select_action(observation)
        action = action.squeeze(0).to("cpu")
        robot.send_action(action)
        dt = time.perf_counter() - t0
        busy_wait(max(0, 1.0 / fps - dt))
    
    use_manual_detection = None
    print("[Inference] Complete.")

################################################################################
# ChatGPT Agent Tools
################################################################################

@tool(return_direct=True)
def reach_the_object():
    """
    Tool to run the inference control loop (i.e. "reach the object") using the policy.
    If a coordinate is already selected, asks the user whether to reselect.
    Then calls detect_target_coords, runs inference, and resets the manual detection flag.
    """
    global policies, robot, rest_position, clicked_coords
    if clicked_coords is not None:
        answer = input(f"Coordinate already selected: {clicked_coords}. Do you want to reselect? (y/n): ").strip().lower()
        if answer == "y":
            detect_target_coords()
    else:
        detect_target_coords()
    config = policies["reach_the_object"]
    run_inference(robot, rest_position, config["chkpt"], config["control_time_s"])
    # Reset manual detection flag and clicked coordinates after inference

    return "Completed reach_the_object inference."

@tool(return_direct=True)
def go_to_home_position_tool():
    """
    Tool to return the robot's arm to the home position.
    """
    global robot, rest_position
    go_to_home_position(robot, rest_position)
    return "Returned to home position."

@tool(return_direct=True)
def detect_target_tool():
    """
    Tool to run target detection.
    """
    coords = detect_target_coords()
    if coords is None:
        return "No target detected."
    else:
        return f"Detected target at {coords}"

@tool(return_direct=True)
def describe_area():
    """Describing what I can see.
    """
    
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    llm_model = "gpt-4o-mini"
    llm = ChatOpenAI(temperature=0.1, model=llm_model, api_key=api_key)

    llm = ChatOpenAI(temperature=0.1, model=llm_model, api_key=api_key)

    observation = robot.capture_observation()
    image_keys = [key for key in observation if "image" in key]
    for key in image_keys:
        if "phone" in key:
            img = cv2.imread(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
    
    
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    _, encoded_img = cv2.imencode('.png', img) 
    base64_img = base64.b64encode(encoded_img).decode("utf-8")    
    
    mime_type = 'image/png'
    encoded_image_url = f"data:{mime_type};base64,{base64_img}"


    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessage(content='Describe in one phrase what objects you see on the table. Not including robot. Start answer with "I see..."'),
            HumanMessagePromptTemplate.from_template(
                 [{'image_url': "{encoded_image_url}", 'type': 'image_url'}],
            )
        ]
    )

    chain = chat_prompt_template | llm
    res = chain.invoke({"encoded_image_url": encoded_image_url})

    return res.content


################################################################################
# Voice Recognition Helper
################################################################################

def listen():
    """
    Listen for a voice command using the microphone.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice command...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"Voice Input: {text}")
        return text
    except Exception as e:
        print("Voice recognition error: " + str(e))
        return None

################################################################################
# Argument Parsing Function
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Robot ChatGPT Agent")
    parser.add_argument("--robot-path", type=str, default="lerobot/configs/robot/koch.yaml",
                        help="Path to the robot configuration file.")
    parser.add_argument("--typing", action="store_true",
                        help="Use typing for commands instead of voice commands.")
    return parser.parse_args()

################################################################################
# Main Function
################################################################################

def main():
    global program_ending, clicked_coords, robot, rest_position, policies

    args = parse_args()
    typing = args.typing

    say("Starting up...", blocking=False)

    print(f"Using robot configuration from: {args.robot_path}")

    #######################################################
    # Setup Robot using configuration file
    #######################################################
    robot_cfg = init_hydra_config(args.robot_path)
    robot = make_robot(robot_cfg)
    robot.connect()

    # Define rest position (could also be read from the robot)
    rest_position = [0.9667969, 128.84766, 174.99023, -16.611328, -4.8339844, 34.716797]

    # Initialize policy configuration
    policies = {
        "reach_the_object": {
            "chkpt": "/home/revolabs/aditya/aditya_lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model",
            "control_time_s": 10
        }
    }

    #######################################################
    # Start Camera Thread
    #######################################################
    cam_thread = threading.Thread(target=camera_thread_func, args=(robot,))
    cam_thread.start()
    time.sleep(2.0)

    #######################################################
    # Create ChatGPT Agent and Tools
    #######################################################
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a helpful robot-arm assistant. Answer super concise."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    llm_model = "gpt-4o-mini"
    llm = ChatOpenAI(temperature=0.1, model=llm_model, api_key=api_key)
    tools_list = [reach_the_object, go_to_home_position_tool, detect_target_tool, describe_area]
    agent = create_tool_calling_agent(llm, tools_list, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)

    #######################################################
    # Chat Loop: Listen for Commands and Call Tools
    #######################################################
    while True:
        if typing:
            command = input("Enter your command (or 'exit' to quit): ")
        else:
            command = listen()
            if command is None:
                continue
        if command.lower() == "exit":
            break
        response = agent_executor.invoke({"input": command})
        print("Response:", response["output"])
        say(response["output"])
    
    #######################################################
    # Cleanup
    #######################################################
    program_ending = True
    cam_thread.join()
    robot.disconnect()
    print("[Main] Program ended gracefully.")

if __name__ == "__main__":
    main()
