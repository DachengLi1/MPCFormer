Main training scripts of MPCFormer. It uses a minimized implementation of Transformer in this directory (i.e. changes in the outmost directory will not affect the behavior here). To reproduce results:

(1) Download Glue data using download_glue_data.py

(2) train a teacher model (pretraining + fine-tuning on downstream tasks) and put in ~/checkpoints/exp/[task_name]. We provide our script in ../text-classification/train_teacher.py.

(3) run MPCFormer distillation process, e.g. for STSB and quad+2quad approximation run:

    python exp2.py --task_name STSB --teacher_dir ~/checkpoints/exp/STSB --student_dir ~/checkpoints/exp/STSB --hidden_act quad --softmax_act 2quad

Potential issues:

(1) In case of dataset minor mismatch, such as "STS-B" and "STSB", please change the data dir as written in task_distill.py. This is due to different naming convention
    of HuggingFace and other Repos.

(2) Hyper-parameters can be overwrite in task_distill.py, 
