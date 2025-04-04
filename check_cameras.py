#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 01:30:24 2024

@author: aadi
"""

import cv2
import os

# Suppress OpenCV warnings/errors in the console
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def list_connected_cameras(max_cameras=50):
    """
    Detects available camera indexes up to max_cameras.
    Returns a list of indexes of working cameras.
    """
    working_cameras = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L)  # Use CAP_V4L to avoid some driver-related errors
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                working_cameras.append(i)
            cap.release()
    
    return working_cameras

def show_camera_one_by_one(camera_indexes):
    """
    Displays each working camera's live feed one by one.
    Press ESC to move to the next camera.
    """
    for i in camera_indexes:
        cap = cv2.VideoCapture(i, cv2.CAP_V4L)
        if not cap.isOpened():
            continue

        print(f"Displaying Camera {i}. Press ESC to move to the next camera.")

        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f'Camera {i}', frame)
            else:
                print(f"Error: Could not read from Camera {i}")
                break

            # Exit current camera feed on ESC key press
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Detecting connected cameras...")
    working_cameras = list_connected_cameras()
    
    if not working_cameras:
        print("No working cameras detected.")
    else:
        print(f"Working cameras: {working_cameras}")
        show_camera_one_by_one(working_cameras)
