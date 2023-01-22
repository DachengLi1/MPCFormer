import os
import subprocess
import json
import shutil
import itertools
import argparse
import torch

exp_name = "dev_app"

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str)
parser.add_argument('--batch', action="store_true")
parser.add_argument('--tune_lr_hidden', type=float, default=5e-5)
parser.add_argument('--tune_lr_pred', type=float, default=1e-5)
parser.add_argument('--tune_bs', type=int, default=32)
parser.add_argument('--tune_softmax', type=str, default=None)
parser.add_argument('--tune_hidden', type=str, default="quad")

args = parser.parse_args()
task_name = args.task_name
batch = args.batch
tune_lr_hidden = args.tune_lr_hidden
tune_lr_pred = args.tune_lr_pred
tune_bs = args.tune_bs
tune_softmax = args.tune_softmax
tune_hidden = args.tune_hidden

base_dir = f"./tmp/{exp_name}/{task_name}/"
if not os.path.exists("./tmp"):
    os.mkdir("./tmp")
if not os.path.exists(f"./tmp/{exp_name}"):
    os.mkdir(f"./tmp/{exp_name}")
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
tinybert_path = "/home/ubuntu/transformers/examples/pytorch/Pretrained-Language-Model/TinyBERT"
log_path = os.path.join(base_dir, "log.txt")
with open(log_path, "w") as f:
    pass

num_devices = torch.cuda.device_count()

def tune_S2():
    assert tune_lr_hidden is not None
    assert tune_lr_pred is not None
    assert tune_bs is not None

    for _ in [0]:
        for _ in [0]:
        # distill hidden layers
            output_dir = os.path.join(base_dir,tune_softmax+"_"+tune_hidden)
            result_path = os.path.join(output_dir, "eval_results.json")
            data_dir = os.path.join("glue_data", task_name)
            teacher_dir = f"exp_{task_name}_t"
            student_dir = f"{exp_name}_{task_name}"
            
            # overwrite config
            config = json.load(open(os.path.join(teacher_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(teacher_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)

            config = json.load(open(os.path.join(student_dir, "config.json")))
            config["log_path"] = log_path 
            config["softmax_act"] = tune_softmax
            config["hidden_act"] = tune_hidden
            print("----------", tune_softmax)
            json_object = json.dumps(config)
            with open(os.path.join(student_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)

            
            if batch:
                cmd = f"python task_distill_batch.py --teacher_model {teacher_dir} \
                       --student_model {student_dir} \
                       --data_dir {data_dir} --task_name {task_name} --output_dir {output_dir} \
                       --max_seq_length 128 --train_batch_size {tune_bs} --learning_rate {tune_lr_hidden}\
                       --do_lower_case --log_path {log_path}"
            
            else:
                cmd = f"python task_distill.py --teacher_model {teacher_dir} \
                       --student_model {student_dir} \
                       --data_dir {data_dir} --task_name {task_name} --output_dir {output_dir} \
                       --max_seq_length 128 --train_batch_size {tune_bs} --learning_rate {tune_lr_hidden}\
                       --do_lower_case --log_path {log_path}"

            subprocess.run(cmd, shell=True)

            # distill pred layers
            config = json.load(open(os.path.join(output_dir, "config.json")))
            config["log_path"] = log_path 
            json_object = json.dumps(config)
            with open(os.path.join(output_dir, "config.json"), "w") as outfile:
                outfile.write(json_object)
            
            output_dir_stage2 = os.path.join(base_dir, tune_softmax+"_"+tune_hidden+"_stage2")
            result_path = os.path.join(output_dir_stage2, "eval_results.json")
            data_dir = os.path.join("glue_data", task_name)
            cmd = f"python task_distill.py --pred_distill  \
                   --teacher_model {teacher_dir} \
                   --student_model {output_dir} \
                   --data_dir {data_dir} \
                   --task_name {task_name} \
                   --output_dir {output_dir_stage2} \
                   --do_lower_case \
                   --learning_rate {tune_lr_pred}  \
                   --num_train_epochs 5 \
                   --eval_step 100 \
                   --max_seq_length 128 \
                   --train_batch_size {tune_bs} --log_path {log_path}"

            subprocess.run(cmd, shell=True)
            with open(log_path, "a") as f:
                f.write(f"distilled S2 with softmax {tune_softmax} lr {str(tune_lr_hidden)} {str(tune_lr_pred)} bs {str(tune_bs)} \n")

tune_S2()

# hold GPU
a = torch.randn(50,50).cuda()
while True:
    a ** 2
