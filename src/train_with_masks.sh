# Heads and MLP mask - Good
export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_heads_mlps/
mkdir -p $MODELS_DIR


for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --head_mask ../masks/heads_mlps/$TASK_NAME/seed_$SEED/head_mask.npy \
        --mlp_mask ../masks/heads_mlps/$TASK_NAME/seed_$SEED/mlp_mask.npy \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

# Heads and MLP mask - Random
# For Importance Pruning we use `mask_mode` and apply that to good mask before training.
export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_heads_mlps_random/
mkdir -p $MODELS_DIR


for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --mask_mode random \ 
        --head_mask ../masks/heads_mlps/$TASK_NAME/seed_$SEED/head_mask.npy \
        --mlp_mask ../masks/heads_mlps/$TASK_NAME/seed_$SEED/mlp_mask.npy \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

# Heads and MLP mask - Bad
# For Importance Pruning we use `mask_mode` and apply that to good mask before training.

export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_heads_mlps_bad/
mkdir -p $MODELS_DIR


for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3, 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --mask_mode bad \
        --head_mask ../masks/heads_mlps_bad/$TASK_NAME/seed_$SEED/head_mask.npy \
        --mlp_mask ../masks/heads_mlps_bad/$TASK_NAME/seed_$SEED/mlp_mask.npy \
        --save_steps 0 \
        --seed $SEED &
    done
    wait;
done


# Heads and MLPS Super Mask - Good
export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_heads_mlps_super/
mkdir -p $MODELS_DIR

for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --head_mask ../masks/heads_mlps_super/$TASK_NAME/seed_$SEED/head_mask.npy \
        --mlp_mask ../masks/heads_mlps_super/$TASK_NAME/seed_$SEED/mlp_mask.npy \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

# Heads and MLPS Super Mask - Bad
export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_heads_mlps_super_bizzaro/
mkdir -p $MODELS_DIR

for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --head_mask ../masks/heads_mlps_super_bizzaro/$TASK_NAME/seed_$SEED/head_mask.npy \
        --mlp_mask ../masks/heads_mlps_super_bizzaro/$TASK_NAME/seed_$SEED/mlp_mask.npy \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

# Heads and MLPS Super Mask - Random
export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_heads_mlps_midling/
mkdir -p $MODELS_DIR

for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --head_mask ../masks/heads_mlps_super_random/$TASK_NAME/seed_$SEED/head_mask.npy \
        --mlp_mask ../masks/heads_mlps_super_random/$TASK_NAME/seed_$SEED/mlp_mask.npy \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

# Magnitude Pruning - Good
export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_global_magnitude_mask/
mkdir -p $MODELS_DIR


for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3, 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --global_mask ../masks/global/$TASK_NAME/seed_$SEED/magnitude_mask.p \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

# Magnitude Pruning - Bad

export MODELS_DIR=../models/finetuned_global_bad_mask/
mkdir -p $MODELS_DIR


for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3, 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --global_mask ../masks/global/$TASK_NAME/seed_$SEED/bad_mask.pt \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

# Magnitude pruning - Random

export MODELS_DIR=../models/finetuned_global_random_mask/
mkdir -p $MODELS_DIR

for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,2 71,3, 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        CUDA_VISIBLE_DEVICES=$GPU python run_glue.py \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --max_seq_length 128 \
        --per_gpu_train_batch_size 16 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-5 \
        --num_train_epochs 3.0 \
        --output_dir $MODELS_DIR/$TASK_NAME/seed_$SEED/ \
        --global_mask ../masks/global/$TASK_NAME/seed_$SEED/random_mask.p \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done
