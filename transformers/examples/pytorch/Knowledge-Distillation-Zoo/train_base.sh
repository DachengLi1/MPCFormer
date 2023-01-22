# example scripts for running various KD methods
# use cifar10, resnet110 (teacher) and resnet20 (student) as examples

# Baseline
CUDA_VISIBLE_DEVICES=6 nohup python -u train_base.py \
                           --save_root "./results/base/" \
                           --data_name cifar100 \
                           --num_class 100 \
                           --net_name mcccnn8 \
			   --act relu \
			   --note base-mcccnn8 > mcccnn8.out &
CUDA_VISIBLE_DEVICES=5 nohup python -u train_base.py \
                           --save_root "./results/base/" \
                           --data_name cifar100 \
                           --num_class 100 \
                           --net_name mcccnn8 \
			   --act quad \
			   --note base-mcccnn8-poly > mcccnn8_poly.out &
CUDA_VISIBLE_DEVICES=3 nohup python -u train_base.py \
                           --save_root "./results/base/" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --net_name mcccnn8 \
			   --act relu \
			   --note base-mcccnn8-10 > mcccnn8-10.out &
CUDA_VISIBLE_DEVICES=4 nohup python -u train_base.py \
                           --save_root "./results/base/" \
                           --data_name cifar10 \
                           --num_class 10 \
                           --net_name mcccnn8 \
			   --act quad \
			   --note base-mcccnn8-poly-10 > mcccnn8_poly-10.out
