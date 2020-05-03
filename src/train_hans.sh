export GLUE_DIR=../data/hans
export MODEL_DIR=../models/finetuned_hans
mkdir -p $MODEL_DIR

# for TASK_NAME in "HANS" "MNLI_TWO" "HANS_MNLI"
# do
#     mkdir -p $MODEL_DIR/$TASK_NAME/
#     for i in 1337,0 42,1 86,2 71,3 166,4
#     do
#         IFS=","; set -- $i;
#         SEED=$1
#         GPU=$2
#         export CUDA_VISIBLE_DEVICES=$GPU
#         python run_glue.py \
#         --model_type bert \
#         --model_name_or_path bert-base-uncased \
#         --task_name $TASK_NAME \
#         --do_train \
#         --do_eval \
#         --do_lower_case \
#         --data_dir $GLUE_DIR/$TASK_NAME \
#         --max_seq_length 128 \
#         --per_gpu_train_batch_size 32 \
#         --learning_rate 2e-5 \
#         --num_train_epochs 3.0 \
#         --output_dir $MODEL_DIR/$TASK_NAME/seed_$SEED/ \
#         --save_steps 0 \
#         --per_gpu_eval_batch_size 64 \
#         --seed $SEED &
#     done
#     wait
# done


for i in 1337,0 42,1 86,2 71,3 166,4
do
    IFS=","; set -- $i;
    SEED=$1
    GPU=$2
    export CUDA_VISIBLE_DEVICES=$GPU
    python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name MNLI_TWO \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/MNLI_TWO_HALF \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir $MODEL_DIR/MNLI_TWO_HALF/seed_$SEED/ \
    --save_steps 0 \
    --per_gpu_eval_batch_size 64 \
    --seed $SEED &
done