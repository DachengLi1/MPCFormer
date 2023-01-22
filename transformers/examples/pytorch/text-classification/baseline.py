import os
import subprocess
import json
import shutil
import itertools
import argparse
import torch

exp_name = "baseline"

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='')
parser.add_argument('--dataset_name', type=str, default='')
parser.add_argument('--model_path', type=str)
parser.add_argument('--baseline_type', type=str)
parser.add_argument('--hidden_act', type=str)
parser.add_argument('--softmax_act', type=str)
args = parser.parse_args()

assert (args.dataset_name and not args.task_name) or (not args.dataset_name and args.task_name)


metric_map = {
    'MNLI': 'eval_accuracy',
    'QQP': 'eval_combined_score',
    'QNLI': 'eval_accuracy',
    'SST2': 'eval_accuracy',
    'RTE': 'eval_accuracy',
    'CoLA': 'eval_matthews_correlation',
    'STSB': 'eval_combined_score',
    'MRPC': 'eval_f1',
    'imdb': 'eval_accuracy',
}


task_name = args.dataset_name if args.dataset_name else args.task_name
new_dataset = True if args.dataset_name else False
model_path = args.model_path
baseline_type = args.baseline_type
hidden_act = args.hidden_act
softmax_act = args.softmax_act

config = json.load(open(os.path.join(model_path, "config.json")))
model_type = config["_name_or_path"]

# sequence length
if task_name == "imdb":
    max_seq_length = 512
else:
    max_seq_length = 128

base_dir = os.path.join("tmp", exp_name, task_name, model_type, hidden_act, softmax_act)
os.makedirs(base_dir, exist_ok=True)

log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "a") as f:
    f.write("new run \n")

num_devices = torch.cuda.device_count()
print(f"num devices: {num_devices}")
lr_list = [1e-6, 5e-6, 1e-5, 1e-4]
if task_name == "imdb":
    bs_list = [32]
else:
    bs_list = [256, 64]
epoch_list = [10, 30, 100]
best = None
best_metric = 0

for lr in lr_list:
    for bs in bs_list:
        for epoch in epoch_list:
            output_dir = os.path.join(base_dir, f"HPO_{baseline_type}" ,str(lr), str(bs), str(epoch))
            result_path = os.path.join(output_dir, "eval_results.json")
            cmd = f"python run_glue.py --model_name_or_path {model_path} \
                   --fp16 --do_train --do_eval --max_seq_length {max_seq_length} \
                  --warmup_ratio 0.2 --per_device_train_batch_size {str(bs//num_devices)} --learning_rate {str(lr)} \
                  --num_train_epochs {epoch} --act {hidden_act} --softmax_act {softmax_act} --output_dir {output_dir} --overwrite_output_dir"
            if baseline_type == "S0":
                cmd += " --scratch"
            if task_name == "imdb":
                cmd += " --dataset_name imdb"
            else:
                cmd += f" --task_name {task_name}"
            subprocess.run(cmd, shell=True)
            result = json.load(open(result_path))
            metric = float(result[metric_map[task_name]])
            if metric > best_metric:
                best = (lr, bs, epoch)
                best_metric = metric
            with open(log_path, "a") as f:
                f.write(f"fine-tuned {baseline_type} with lr {str(lr)} bs {str(bs)} epoch {epoch}, acc: {metric} \n")

best_lr, best_bs, best_epoch = best
with open(log_path, "a") as f:
    f.write(f"best {baseline_type} with lr {best_lr} bs {best_bs} epoch {best_epoch}, acc: {best_metric} \n")
