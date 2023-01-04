import os
import subprocess
import json
import shutil
import itertools
import argparse
import torch

exp_name = "exp"

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
parser.add_argument('--metric_name', type=str)
parser.add_argument('--model_name', type=str)

args = parser.parse_args()
task_name = args.task_name
metric_name = args.metric_name
model_name = args.model_name

base_dir = f"./tmp/{exp_name}/{task_name}/{model_name}"
if not os.path.exists("./tmp"):
    os.mkdir("./tmp")
if not os.path.exists(f"./tmp/{exp_name}"):
    os.mkdir(f"./tmp/{exp_name}")
if not os.path.exists(f"./tmp/{exp_name}/{task_name}"):
    os.mkdir(f"./tmp/{exp_name}/{task_name}")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("new run \n")

num_devices = torch.cuda.device_count()

def HPO_teacher_bert():
    lr_list = [2e-5, 3e-5, 4e-5, 5e-5]
    bs = 32 // num_devices
    teacher_acc = []
    
    for lr in lr_list:
        output_dir = os.path.join(base_dir, str(lr))
        result_path = os.path.join(output_dir, "eval_results.json")
        cmd = f"python run_glue.py --model_name_or_path bert-base-uncased --task_name {task_name} \
              --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {bs} --learning_rate {str(lr)} \
              --num_train_epochs 3 --save_steps 200 --act gelu --softmax_act softmax --output_dir {output_dir} --overwrite_output_dir"
        subprocess.run(cmd, shell=True)
        result = json.load(open(result_path))
        acc = float(result[metric_name])
        teacher_acc.append(acc)
        with open(log_path, "a") as f:
            f.write(f"fine-tuned Bert base with lr {str(lr)}, acc: {acc} \n")

    max_acc = max(teacher_acc)
    best_lr = lr_list[teacher_acc.index(max_acc)]
    with open(log_path, "a") as f:
        f.write(f"best teacher with lr {best_lr}, acc: {max_acc} \n")

def HPO_teacher_bert_large():
    lr_list = [2e-5, 3e-5, 4e-5, 5e-5]
    bs = 32 // num_devices
    teacher_acc = []
    
    for lr in lr_list:
        output_dir = os.path.join(base_dir, str(lr))
        result_path = os.path.join(output_dir, "eval_results.json")
        cmd = f"python run_glue.py --model_name_or_path bert-large-uncased --task_name {task_name} \
              --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size {bs} --learning_rate {str(lr)} \
              --num_train_epochs 3 --save_steps 200 --act gelu --softmax_act softmax --output_dir {output_dir} --overwrite_output_dir"
        subprocess.run(cmd, shell=True)
        result = json.load(open(result_path))
        acc = float(result[metric_name])
        teacher_acc.append(acc)
        with open(log_path, "a") as f:
            f.write(f"fine-tuned Bert large with lr {str(lr)}, acc: {acc} \n")

    max_acc = max(teacher_acc)
    best_lr = lr_list[teacher_acc.index(max_acc)]
    with open(log_path, "a") as f:
        f.write(f"best teacher with lr {best_lr}, acc: {max_acc} \n")

def HPO_teacher_roberta():
    lr_list = [2e-5, 3e-5, 4e-5]
    bs_list = [16, 32]# // num_devices
    teacher_acc = []
    
    for total_bs in bs_list:
        bs = total_bs // num_devices
        for lr in lr_list:
            output_dir = os.path.join(base_dir, str(lr), str(total_bs))
            result_path = os.path.join(output_dir, "eval_results.json")
            cmd = f"python run_glue.py --model_name_or_path roberta-base --task_name {task_name} \
                  --do_train --do_eval --warmup_ratio 0.06 --max_seq_length 128 --per_device_train_batch_size {bs} --learning_rate {str(lr)} \
                  --num_train_epochs 10 --save_steps 200 --act gelu --softmax_act softmax --output_dir {output_dir} --overwrite_output_dir"
            subprocess.run(cmd, shell=True)
            result = json.load(open(result_path))
            acc = float(result[metric_name])
            teacher_acc.append(acc)
            with open(log_path, "a") as f:
                f.write(f"fine-tuned roBerta base with lr {str(lr)} bs {str(total_bs)}, acc: {acc} \n")

    max_acc = max(teacher_acc)
    best_lr = lr_list[teacher_acc.index(max_acc)]
    with open(log_path, "a") as f:
        f.write(f"best teacher with lr {best_lr}, acc: {max_acc} \n")

if model_name == "bert-base-uncased":
    HPO_teacher_bert()
if model_name == "bert-large-uncased":
    HPO_teacher_bert_large()
elif model_name == "roberta-base":
    HPO_teacher_roberta()
