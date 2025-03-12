#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:04:02 2025

@author: aadi
"""

import os
import cv2
import time
import torch
import platform
import threading
from PIL import Image
from typing import List
import numpy as np
import base64
from dotenv import load_dotenv, find_dotenv
import argparse
import math
import getpass

# For generative AI bounding-box detection (if needed)
import google.generativeai as genai

from pynput.keyboard import Key, Listener

# Robot-specific imports
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

# Policy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect

# LangChain / ChatGPT Agent imports
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
target_coords = None    # Will hold (x, y) or None
use_manual_detection = False  # Global flag for manual detection
angle = 0

# Global to store if we are in typing mode (default is False)
typing_mode = False

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
    # """
    # Simple TTS wrapper that works on macOS, Linux, or Windows.
    # """
    # if platform.system() == "Darwin":
    #     cmd = f'say "{text}"'
    # elif platform.system() == "Linux":
    #     cmd = f'spd-say "{text}"'
    # elif platform.system() == "Windows":
    #     cmd = (
    #         'PowerShell -Command "Add-Type -AssemblyName System.Speech; '
    #         f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
    #     )
    # else:
    #     print(f"(say) OS not recognized. Printing text: {text}")
    #     return

    # if not blocking and platform.system() in ["Darwin", "Linux"]:
    #     cmd += " &"
    # os.system(cmd)
    pass

def update_angle(key):
    global angle
    if key == Key.left:
        angle -= 2
    elif key == Key.right:
        angle += 2

def put_the_marker(image: np.ndarray, coords: tuple, radius=10,
                   border_color=(0, 0, 255), cross_color=(0, 0, 255),
                   bg_color=(255, 255, 255)):
    """
    Draw a marker on the given image at the specified `coords`.
    """
    if coords is None:
        return image
    global angle
    x, y = coords
    center = (x, y)
    cv2.circle(image, center, radius, bg_color, -1)
    cv2.circle(image, center, radius, border_color, 2)
    cv2.line(image, center,
             (x - int(math.sin(math.radians(angle)) * radius),
              y + int(math.cos(math.radians(angle)) * radius)),
             cross_color, 2)
    cv2.line(image, center,
             (x - int(math.cos(math.radians(angle)) * radius),
              y - int(math.sin(math.radians(angle)) * radius)),
             cross_color, 2)
    cv2.line(image, center,
             (x + int(math.cos(math.radians(angle)) * radius),
              y + int(math.sin(math.radians(angle)) * radius)),
             cross_color, 2)
    cv2.line(image, center,
             (x + int(math.sin(math.radians(angle)) * radius),
              y - int(math.cos(math.radians(angle)) * radius)),
             cross_color, 2)
    cv2.arrowedLine(image,
                    (x + int(math.cos(math.radians(angle)) * radius),
                     y + int(math.sin(math.radians(angle)) * radius)),
                    (x + int(math.cos(math.radians(angle)) * 25),
                     y + int(math.sin(math.radians(angle)) * 25)),
                    cross_color, 4, tipLength=0.75)
    return image

def on_mouse_double_click(event, x, y, flags, param):
    """
    Mouse callback used in manual mode.
    """
    global target_coords, use_manual_detection
    if use_manual_detection and event == cv2.EVENT_LBUTTONDBLCLK:
        target_coords = (x, y)
        print(f"[Manual Detection] Double-click at: {target_coords}")

def object_detection(image: np.ndarray, object_to_detect: str) -> List[tuple]:
    """
    Uses generative AI (via google.generativeai) to detect bounding boxes for the
    given `object_to_detect` in `image`. Returns a list of centroids as (x, y).
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
    Continuously reads frames from the robot's "phone" camera, updates `img_buffer`,
    and displays the frame.
    """
    global program_ending, target_coords, use_manual_detection
    cv2.namedWindow("phone", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("phone", 640, 480)
    cv2.setMouseCallback("phone", on_mouse_double_click)
    while not program_ending:
        frame = robot.cameras["phone"].async_read()
        if frame is not None:
            img_buffer["phone"] = frame.copy()
            display_frame = cv2.cvtColor(img_buffer["phone"], cv2.COLOR_RGB2BGR)
            if target_coords is not None:
                put_the_marker(display_frame, target_coords)
            cv2.imshow("phone", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            program_ending = True
            break
    cv2.destroyAllWindows()
    print("[Camera Thread] Exiting camera loop.")



def detect_target_coords(mode="ai", object_to_detect="object") -> tuple:
    """
    Detects the target coordinate.
    In 'manual' mode, uses the previous logic: instructs you to double-click on the phone window,
    then waits for you to press ENTER (or provide an empty response) to confirm.
    In 'ai' mode, uses AI-based detection and, if multiple objects are found, asks which index to use.
    """
    global target_coords, img_buffer, use_manual_detection, typing_mode
    if mode.lower() == "manual":
        use_manual_detection = True
        say("Please double-click on the phone window to mark your object. Then press ENTER when done selecting points.", blocking=True)
        print("Manual detection: Please double-click on the phone window to mark your object and press enter here.")
        # The prompt is intentionally left empty so nothing is shown.
        getpass.getpass(prompt='')

        # _ = get_command(typing_mode)  # Wait for confirmation (ENTER)
        if target_coords is None:
            print("[Manual Detection] No point was clicked.")
            return None
        return target_coords
    # Default to AI detection.
    frame_phone = img_buffer["phone"]
    coords_list = []
    if frame_phone is not None:
        frame_bgr = cv2.cvtColor(frame_phone, cv2.COLOR_RGB2BGR)
        coords_list = object_detection(frame_bgr, object_to_detect)
        print("[AI Detection] Detected coordinates:", coords_list)
    else:
        print("[AI Detection] No phone frame in buffer!")
    if not coords_list:
        target_coords = None
        return None
    if len(coords_list) > 1:
        print("Multiple objects found:")
        for idx, coord in enumerate(coords_list):
            print(f"Index {idx}: {coord}")
        print("Please specify which object index to select:")
        chosen_index_str = get_command(typing_mode)
        try:
            chosen_index = int(chosen_index_str.strip())
        except Exception as e:
            print("Invalid index provided, defaulting to index 0.")
            chosen_index = 0
        if chosen_index < 0 or chosen_index >= len(coords_list):
            print("Index out of range, defaulting to index 0.")
            chosen_index = 0
        target_coords = coords_list[chosen_index]
        return target_coords
    else:
        target_coords = coords_list[0]
        return target_coords

def go_to_home_position(robot, rest_position):
    """
    Moves the follower arm from its current position to `rest_position` in small increments.
    """
    current_pos = robot.follower_arms['main'].read("Present_Position")
    steps = 30
    for i in range(1, steps + 1):
        alpha = i / steps
        intermediate_pos = current_pos + alpha * (rest_position - current_pos)
        robot.follower_arms['main'].write("Goal_Position", intermediate_pos)
        time.sleep(0.05)

def grip_the_object():
    global target_coords
    print("grip the object at", target_coords)

def early_stop_robot(current_angles: torch.Tensor, tolerance: float = 5,
                     stable_duration_seconds: float = 3.0, fps: int = 30) -> bool:

    # Initialize the history list on the first call.
    if not hasattr(early_stop_robot, "history"):
        early_stop_robot.history = []
    
    # Append the current angles (cloned to avoid mutation issues)
    early_stop_robot.history.append(current_angles.clone())
    
    # Calculate the number of frames that correspond to the desired duration.
    required_frames = int(stable_duration_seconds * fps)
    
    # If we don't yet have enough frames, return False.
    if len(early_stop_robot.history) < required_frames:
        return False
    
    # Compare the angles from the frame at time t (start of window)
    # and the current frame (time t + stable_duration_seconds)
    start_angles = early_stop_robot.history[-required_frames]
    end_angles = current_angles  # Most recent frame
    
    # Calculate absolute difference for each joint.
    diff = torch.abs(end_angles - start_angles)
    
    # If all differences are below the tolerance, we consider it stable.
    if torch.all(diff < tolerance):
        print("Early stopping")
        early_stop_robot.history = []  # Clear the history for future use.
        return True
    else:
        return False

def run_inference(robot, rest_position, chkpt, control_time_s):
    """
    Loads the policy from the given checkpoint and runs a control loop for `control_time_s` seconds.
    The robot captures an observation, marks the selected target (if any), computes an action using the policy,
    and sends the action to the robot.
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
                if target_coords is not None and "phone" in name:
                    img_bgr = cv2.cvtColor(observation[name].numpy(), cv2.COLOR_RGB2BGR)
                    put_the_marker(img_bgr, target_coords)
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    observation[name] = torch.from_numpy(img_rgb)
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0).to(device)
        action = policy.select_action(observation)
        action = action.squeeze(0).to("cpu")
        if early_stop_robot(action,fps=fps) == True:
            break
        robot.send_action(action)
        dt = time.perf_counter() - t0
        busy_wait(max(0, 1.0 / fps - dt))
    
    use_manual_detection = False
    print("[Inference] Complete.")
    
def is_exit_command(command: str) -> bool:
    """
    Uses ChatGPT to determine if the given command is an exit instruction.
    The prompt instructs the model to answer only 'yes' or 'no'.
    """
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    classifier = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
    prompt = f"""You are a classifier. Determine if the following command indicates that the user wants to exit the session. Answer only 'yes' or 'no'.
Command: "{command}"
"""
    response = classifier.invoke(prompt)
    return "yes" in str(response).lower()

################################################################################
# ChatGPT Agent Tools
################################################################################

@tool(return_direct=True)
def reach_the_object(object_prompt: str = "object"):
    """
    Runs the inference control loop to reach the target object using AI-based detection.
    """
    global policies, robot, rest_position, target_coords
    detect_target_coords(mode="ai", object_to_detect=object_prompt)
    config = policies["reach_the_object"]
    run_inference(robot, rest_position, config["chkpt"], config["control_time_s"])
    return "Completed reach_the_object inference (AI mode)."

@tool(return_direct=True)
def reach_the_object_manual():
    """
    In manual mode, instructs you to double-click on the phone window to select the target.
    After pressing ENTER when done, runs the inference control loop to reach the target.
    """
    global policies, robot, rest_position, target_coords
    coords = detect_target_coords(mode="manual", object_to_detect="object")
    if coords is None:
        return "No target selected manually."
    config = policies["reach_the_object"]
    run_inference(robot, rest_position, config["chkpt"], config["control_time_s"])
    return "Completed reach_the_object inference (Manual mode)."

@tool(return_direct=True)
def grip_the_object_tool():
    """If a command such as 'grip the object', 'pick up the object', or 'pickup the object' is received, where object or mode is not specified just call grip tool
    else, first call the 'reach_the_object' tool (or 'reach_the_object_manual' if manual mode is specified) to move toward the object, 
    then call the 'grip_the_object_tool' to grasp it."""
   
    grip_the_object()
    return "Gripped the object."


@tool(return_direct=True)
def go_to_home_position_tool():
    """
    Returns the robot's arm to the home position.
    """
    global robot, rest_position,target_coords
    go_to_home_position(robot, rest_position)
    target_coords = None
    return "Returned to home position."

@tool(return_direct=True)
def detect_target_tool(object_prompt: str = "object"):
    """
    Runs target detection using AI and reports the detected coordinates.
    """
    coords = detect_target_coords(mode="ai", object_to_detect=object_prompt)
    if coords is None:
        return "No target detected."
    else:
        return f"Detected target at {coords}"

@tool(return_direct=True)
def describe_area():
    """
    Captures the current observation and uses a ChatGPT-like model to describe the area in one phrase.
    """
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")
    llm_model = "gpt-4o-mini"
    llm = ChatOpenAI(temperature=0.1, model=llm_model, api_key=api_key)

    observation = robot.capture_observation()
    phone_image = None
    for key in observation:
        if "image" in key and "phone" in key:
            img = observation[key].numpy()
            phone_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            break

    if phone_image is None:
        return "No phone image available for description."

    _, encoded_img = cv2.imencode('.png', phone_image) 
    base64_img = base64.b64encode(encoded_img).decode("utf-8")    
    mime_type = 'image/png'
    encoded_image_url = f"data:{mime_type};base64,{base64_img}"

    chat_prompt_template = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessage(content='Describe in one phrase what objects you see on the table. Exclude the robot. Start your answer with "I see..."'),
            HumanMessagePromptTemplate.from_template(
                 [{'image_url': "{encoded_image_url}", 'type': 'image_url'}],
            )
        ]
    )

    chain = chat_prompt_template | llm
    res = chain.invoke({"encoded_image_url": encoded_image_url})
    return res.content

################################################################################
# Voice / Typing Command Helper
################################################################################

def listen_voice():
    """
    Listen for a voice command using the microphone.
    """
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice command...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"Voice Input: {text}")
        return text
    except Exception as e:
        print("Voice recognition error:", str(e))
        return None

def get_command(use_typing: bool):
    """
    Returns the command from either keyboard (if use_typing is True) or voice.
    """
    if use_typing:
        command = input("Enter your command (or response): ")
        return command
    else:
        return listen_voice()

################################################################################
# Main Function (Runtime Command Processing)
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous Robot ChatGPT Agent (Runtime Command Mode)")
    parser.add_argument("--robot-path", type=str, default="lerobot/configs/robot/koch.yaml",
                        help="Path to the robot configuration file.")
    parser.add_argument("--typing", action="store_true",
                        help="Use keyboard input instead of voice commands.")
    return parser.parse_args()

def main():
    global program_ending, target_coords, robot, rest_position, policies, typing_mode

    args = parse_args()
    typing_mode = args.typing
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
            "chkpt": "/home/revolabs/cs_capstone/lerobot/outputs/train/act_koch_reach_the_marker/pretrained_model",
            "control_time_s": 30
        }
    }

    #######################################################
    # Start Camera Thread
    #######################################################
    cam_thread = threading.Thread(target=camera_thread_func, args=(robot,))
    cam_thread.start()
    time.sleep(2.0)
    
    listener_angle = Listener(on_press=update_angle)
    listener_angle.start()

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
    tools_list = [reach_the_object, reach_the_object_manual, grip_the_object_tool,go_to_home_position_tool, detect_target_tool, describe_area]
    agent = create_tool_calling_agent(llm, tools_list, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=True)

    #######################################################
    # Runtime Command Loop (Voice or Typing-based)
    #######################################################
    print("Entering runtime command loop. Say or type 'exit' to quit.")
    while True:
        if typing_mode:
            command = input("Enter your command (or response): ")
        else:
            command = listen_voice()
        if command is None:
            continue
        if is_exit_command(command):
            break
        response = agent_executor.invoke({"input": command})
        output = response.get("output", "")
        print("Response:", output)
        say(output)
    
    #######################################################
    # Cleanup
    #######################################################
    program_ending = True
    cam_thread.join()
    robot.disconnect()
    print("[Main] Program ended gracefully.")

if __name__ == "__main__":
    main()
