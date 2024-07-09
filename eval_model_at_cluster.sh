
ds=$1
fold=$2
model=$3

rm /Users/dyxa68da/PycharmProjects/tabtrans/non_code/pretraining_checkpoints/*

scp alex.nhr.fau.de:"/home/hpc/v101be/v101be10/experiments/tabtrans/output/no_init_$3/$1_$2/checkpoints/model_*" /Users/dyxa68da/PycharmProjects/tabtrans/non_code/pretraining_checkpoints

python -m main_CV_experiment --dataset_id $1 --outer_fold $2 --num_layers 10 --dim 128 --learning_rate 0.001 --num_epochs 0 --num_tables_per_batch 8 --pretrain_checkpoints_dir /Users/dyxa68da/PycharmProjects/tabtrans/non_code/pretraining_checkpoints --model_type $3
