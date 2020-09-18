export GLUE_DIR=../data/glue
export CUDA_VISIBLE_DEVICES=0
mkdir -p ../models/finetuned

for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    mkdir -p ../experiments/$TASK_NAME/
    for SEED in 1337 42 86 71 166
    do
        python run_glue.py \
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
        --output_dir ../models/finetuned/$TASK_NAME/seed_$SEED/ \
        --save_steps 0 \
        --eval_batch_size 64 \
        --seed $SEED
    done
done
