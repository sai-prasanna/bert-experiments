export GLUE_DIR=../data/glue
export MODELS_DIR=../models/finetuned_magnitude_mask/
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


export MODELS_DIR=../models/finetuned_random_mask/
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

export MODELS_DIR=../models/finetuned_random_mimic_size/
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
        --global_mask ../masks/global/$TASK_NAME/seed_$SEED/random_mimic.pt \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done

export MODELS_DIR=../models/finetuned_magnitude_mimic_size/
mkdir -p $MODELS_DIR

for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,0 42,1 86,3 71,4 166,4
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
        --global_mask ../masks/global/$TASK_NAME/seed_$SEED/magnitude_mimic.pt \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done


export MODELS_DIR=../models/finetuned_global_bad_mimic_size/
mkdir -p $MODELS_DIR

for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 1337,5 42,6 86,7
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
        --global_mask ../masks/global/$TASK_NAME/seed_$SEED/bad_mimic.pt \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done
for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p $MODELS_DIR/$TASK_NAME/
    for i in 71,5 166,6
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
        --global_mask ../masks/global/$TASK_NAME/seed_$SEED/bad_mimic.pt \
        --save_steps 0 \
        --seed $SEED &
    done
    wait
done
