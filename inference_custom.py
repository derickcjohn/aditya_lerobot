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
from PIL import Image
import google.generativeai as genai
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
        "phone": OpenCVCamera("/dev/video12", fps=30, width=640, height=480),
        "laptop": OpenCVCamera("/dev/video10", fps=30, width=640, height=480),
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

import cv2

def capture_all_coordinates(robot):
        """
        Captures a single coordinate (x, y) for each image in the robot observation 
        by asking the user to double-click on the displayed images.
        
        Returns:
        list: A list of (x, y) tuples representing the selected coordinates.
        """
        coords_list = []
        
        # 1) Capture the current state/observation from the robot
        observation = robot.capture_observation()
        
        # 2) Find all keys that contain 'image'
        image_keys = [key for key in observation if "phone" in key]
        
        # 3) For each image key, capture a coordinate
        for key in image_keys:
            # Convert from torch Tensor (RGB) to NumPy (BGR) if needed
            img_rgb = observation[key].numpy()  # shape (H, W, 3) or (3, H, W)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
            print(f"Please double-click on the window to select a point for '{key}'")
            coords = cursor_coordinate_input(img_bgr, window_name=str(key))
        
            # If user pressed Esc without double-click, coords will be None
            if coords is None:
                print(f"No coordinates captured for '{key}' (user pressed Esc).")
            else:
                print(f"Coordinates for '{key}': {coords}")
                coords_list.append(coords)
        
        # 4) Print and return the list of coordinates
        print("All coordinates:", coords_list)
        return coords_list


def cursor_coordinate_input(image, window_name=None):
        """
        Displays the image, waits for a double-click from the user,
        and returns the (x, y) coordinates of the double-click point in the image.
        
        Returns:
        (x, y) coordinates as a tuple of integers if a double-click happens;
        None if the user exits without a double-click.
        """
        
        if image is None:
            print("Error: Cannot load image.")
            return None
        
        coord_container = {'coords': None}
        
        # Define the mouse callback
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                param['coords'] = (x, y)
        
        # Create a named window and set the mouse callback
        window_name = window_name + " Double-Click to Capture Coordinates"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse, coord_container)
        
        # Display the image and wait for a double-click or Esc key
        while True:
            cv2.imshow(window_name, image)
            key = cv2.waitKey(1) & 0xFF
        
            if coord_container['coords'] is not None:
                break
            if key == 27:  # Esc key
                break
        
        cv2.destroyAllWindows()
        return coord_container['coords']



def put_the_marker(
    image,
    coords_list,
    index=0,
    radius=10,
    border_color=(0, 0, 255),
    cross_color=(0, 0, 255),
    bg_color=(255, 255, 255)
):
        """
        Draws a circle with a white background, red border, and a red cross in the center
        at the coordinates corresponding to the provided `index` in `coords_list`.
        
        Args:
        image: The frame/image (NumPy array) to draw on.
        index: The index to look up in coords_list.
        coords_list: A list of (x, y) coordinates.
        radius: Radius of the circle.
        border_color: BGR tuple for the circle's border color (default: red).
        cross_color: BGR tuple for the cross color (default: red).
        bg_color: BGR tuple for the circle's background color (default: white).
        
        The function does nothing if `index` is out of range.
        """
        
        if index >= len(coords_list) or index < 0:
            print(f"put_the_marker: Index {index} is out of range.")
            return
        
        center = coords_list[index]
        if center is None:
            print(f"put_the_marker: Index {index} has no coordinates (None).")
            return
        
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

    
def object_detection(robot,object_to_detect):
    
        GOOGLE_API_KEY = "AIzaSyBzC0kdp5WKkhaHQVWvqpFdvoCGCvdyCIE"
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # 1) Capture the current state/observation from the robot
        observation = robot.capture_observation()
        
        # 2) Find all keys that contain 'image'
        image_keys = [key for key in observation if "phone" in key]
        
        # 3) For each image key, capture a coordinate
        for key in image_keys:
            # Convert from torch Tensor (RGB) to NumPy (BGR) if needed
            img_rgb = observation[key].numpy()  # shape (H, W, 3) or (3, H, W)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
        # Load the image (both via PIL and cv2)
        # pil_img = Image.open(image_path)
        cv_img = img_bgr
        pil_img = Image.fromarray(img_rgb)  # Convert OpenCV image to PIL format
        
        # Get height and width of the original image
        height, width, _ = cv_img.shape  # shape -> (rows = height, cols = width, channels)
        
        # Choose a Gemini model
        model = genai.GenerativeModel(model_name="learnlm-1.5-pro-experimental")
        
        # Prompt to return bounding boxes in [y_min, x_min, y_max, x_max] format
        # prompt = "Return a bounding box for position where usb is inserted to laptop in this image in [y_min, x_min, y_max, x_max] format."
        prompt=f"""Please detect {object_to_detect} in this image. For each object, output:
         
        1. The object label created (e.g., "phone", "cable").
        2. Its bounding box in [y_min, x_min, y_max, x_max] format.
    
        Use the exact format:
          label:[y_min, x_min, y_max, x_max]
    
        One object per line, with no additional heading, text or explanation."""
    
        response = model.generate_content([pil_img, prompt])
        # print("Raw model response:\n", response.text)
        
        # Dictionary: { label: (centroid_x, centroid_y) }
        coords_list = []  
    
        lines = response.text.strip().split('\n')
    
        for line in lines:
            line = line.strip()
            if not line:
                continue
    
            if ':' not in line:
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
    
            # cv2.circle(cv_img, (centroid_x, centroid_y), 10, (0, 0, 255), -1)
    
            coords_list.append((centroid_x, centroid_y))  
        
        coords_list.sort(key=lambda coord: coord[0])
        
        # Annotate the image with indices
        for idx, (centroid_x, centroid_y) in enumerate(coords_list):
            cv2.putText(cv_img, str(idx), (centroid_x, centroid_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("\nDetection Results (Centroids):")
        print(coords_list)
        
        print("check the image & press esc to continue")
        cv2.imshow("Centroids", cv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        return coords_list
    

manual_detection = True
index=0

if manual_detection:
    coords_list = capture_all_coordinates(robot)
else:
    # object_to_detect = "blue object"
    print("enter your prompt")
    object_to_detect = input("prompt :")
    coords_list=object_detection(robot, object_to_detect)
    if len(coords_list)>1:
        print("more than 1 point if interests")
        index = int(input("enter the object number :"))



inference_time_s = 10
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
                
                if coords_list is not None and "phone" in name:
                    img_bgr = cv2.cvtColor(observation[name].numpy(), cv2.COLOR_RGB2BGR)
                    # OpenCV expects (x, y) as (center_x, center_y)
                    put_the_marker(img_bgr,coords_list,index)
                    img_rgb = cv2.cvtColor(observation[name].numpy(), cv2.COLOR_BGR2RGB)
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