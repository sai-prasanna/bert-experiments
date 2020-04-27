export GLUE_DIR=../data/glue
export CUDA_VISIBLE_DEVICES=2
export MODELS_DIR=../finetuned_models_with_head_and_mlp_mask_bad/
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
        --head_mask ../head_mlp_masks_results/$TASK_NAME/seed_$SEED/head_mask.npy \
        --mlp_mask ../head_mlp_masks_results/$TASK_NAME/seed_$SEED/mlp_mask.npy \
        --mask_mode bad \
        --save_steps 0 \
        --seed $SEED &
    done
    wait;
done