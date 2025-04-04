#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 01:34:41 2024

@author: aadi
"""
"""
python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --num-episodes 10 \
  -p "/home/revolabs/aditya/aditya_lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model"
  
  
  python lerobot/scripts/control_robot.py record \
     --robot-path lerobot/configs/robot/koch.yaml \
     --fps 30 \
     --root data \
     --repo-id ${HF_USER}/eval_koch_reach_the_object \
     --tags tutorial eval \
     --warmup-time-s 5 \
     --episode-time-s 10 \
     --reset-time-s 30 \
     --num-episodes 1 \
     -p "/home/revolabs/aditya/aditya_lerobot/outputs/train/act_koch_reach_the_object/checkpoints/last/pretrained_model"
 
 
 python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/revobots.yaml \
  --fps 30 \
  --root data \
  --repo-id koch_big_robot \
  --tags tutorial \
  --warmup-time-s 300 \
  --episode-time-s 300 \
  --reset-time-s 300 \
  --num-episodes 2 \
  --push-to-hub 0
  
  
  DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/koch_reach_the_object \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/act_koch_reach_the_object \
  hydra.job.name=act_koch_test \
  device=cuda
  
  cd aditya/aditya_lerobot
  python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/revobots.yaml
  
  python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml

python lerobot/scripts/control_robot.py replay     --fps 30     --root data     --repo-id koch_big_robot     --episode 1 --robot-path lerobot/configs/robot/revobots.yaml
"""