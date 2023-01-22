#!/bin/basCUDA_VISIBLE_DEVICES=2,3 nohup python exp2.py --task_name MRPC --bs 8 --teacher_dir ~/checkpoints/exp/MRPC --student_dir ~/checkpoints/exp/MRPC --hidden_act quad --softmax_act 2quad &
CUDA_VISIBLE_DEVICES=4 nohup python exp2.py --task_name STSB --lr_hidden 3e-5 --bs 32 --teacher_dir ~/checkpoints/exp/STSB --student_dir ~/checkpoints/exp/STSB --hidden_act quad --softmax_act 2quad &
CUDA_VISIBLE_DEVICES=5 nohup python exp2.py --task_name STSB --lr_hidden 3e-5 --bs 8 --teacher_dir ~/checkpoints/exp/STSB --student_dir ~/checkpoints/exp/STSB --hidden_act quad --softmax_act 2quad &
CUDA_VISIBLE_DEVICES=2,3 nohup python exp2.py --task_name QQP  --teacher_dir ~/checkpoints/exp/QQP --student_dir ~/checkpoints/exp/QQP --hidden_act quad --softmax_act 2quad &
#CUDA_VISIBLE_DEVICES=5 nohup python exp2.py --task_name RTE --bs 8 --teacher_dir ~/checkpoints/exp/RTE --student_dir ~/checkpoints/exp/RTE --hidden_act quad --softmax_act 2quad &
