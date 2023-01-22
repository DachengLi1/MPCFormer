#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 nohup python baseline.py --task_name RTE --model_path ~/checkpoints/exp/RTE --baseline_type S0 --hidden_act quad --softmax_act 2quad &
CUDA_VISIBLE_DEVICES=6,7 nohup python baseline.py --task_name CoLA --model_path ~/checkpoints/exp/CoLA --baseline_type S0 --hidden_act quad --softmax_act 2quad

