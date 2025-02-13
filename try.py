#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:35:35 2025

@author: aadi
"""
import os
import sys
sys.path.append("//home/revolabs/aditya/GroundingDINO")
os.environ["CUDA_VISIBLE_DEVICES"]="0"

CONFIG_PATH = "/home/revolabs/aditya/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
WEIGHTS_PATH = "/home/revolabs/aditya/GroundingDINO/weights/groundingdino_swint_ogc.pth"
from groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(CONFIG_PATH, WEIGHTS_PATH)
IMAGE_PATH = "/home/revolabs/Downloads/phone.jpg"

TEXT_PROMPT = "blue object"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
