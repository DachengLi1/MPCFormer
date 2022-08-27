# example scripts for running various KD methods
# use cifar10, resnet110 (teacher) and resnet20 (student) as examples

# SoftTarget
CUDA_VISIBLE_DEVICES=0 python -u train_kd.py \
                           --save_root "./results/st/" \
                           --t_model "./results/base/base-cnn8/model_best.pth.tar" \
                           --s_init "./results/base/base-cnn8-poly/initial_r8_poly.pth.tar" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --t_name mcccnn8 \
                           --s_name mcccnn8_poly \
                           --kd_mode st \
                           --lambda_kd 0.1 \
                           --T 4.0 \
                           --note st-c10-cnn8-cnn8_poly

