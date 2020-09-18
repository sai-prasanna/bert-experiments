export GLUE_DIR=../data/glue
mkdir -p ../masks/

# Heads Only
for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    echo "Masking" $TASK_NAME
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        OUTPUT_DIR=../masks/heads/$TASK_NAME/seed_$SEED/
        mkdir -p $OUTPUT_DIR
        export CUDA_VISIBLE_DEVICES=$GPU
        python find_head_masks.py \
                --model_type bert \
                --model_name_or_path ../models/finetuned/$TASK_NAME/seed_$SEED/ \
                --task_name $TASK_NAME \
                --data_dir $GLUE_DIR/$TASK_NAME \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --batch_size 64 \
                --try_masking &
    done
    wait;
done

# MLPS Only
export GLUE_DIR=../data/glue
for TASK_NAME in  "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    echo "Masking" $TASK_NAME
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        OUTPUT_DIR=../masks/mlps/$TASK_NAME/seed_$SEED/
        mkdir -p $OUTPUT_DIR
        CUDA_VISIBLE_DEVICES=$GPU python find_mlp_masks.py \
                --model_type bert \
                --model_name_or_path ../models/finetuned/$TASK_NAME/seed_$SEED/ \
                --task_name $TASK_NAME \
                --data_dir $GLUE_DIR/$TASK_NAME \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --batch_size 64 \
                --try_masking &
    done
    wait;
done

# Heads and MLPS
export GLUE_DIR=../data/glue
for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE" "WNLI"
do
    echo "Masking" $TASK_NAME
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        OUTPUT_DIR=../masks/heads_mlps/$TASK_NAME/seed_$SEED/
        mkdir -p $OUTPUT_DIR
        export CUDA_VISIBLE_DEVICES=$GPU
        python find_head_mlp_masks.py \
                --model_type bert \
                --model_name_or_path ../models/finetuned/$TASK_NAME/seed_$SEED/ \
                --task_name $TASK_NAME \
                --data_dir $GLUE_DIR/$TASK_NAME \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --batch_size 64 \
                --save_mask_all_iterations \  # For computing importance score distributions
                --try_masking &
    done
    wait;
done

# Magnitude Pruning
export GLUE_DIR=../data/glue
for TASK_NAME in "CoLA" "SST-2" "MRPC" "STS-B" "QQP" "MNLI" "QNLI" "RTE"
do
    echo "Masking" $TASK_NAME
    for i in 1337,0 42,1 86,2 71,3 166,4
    do
        IFS=","; set -- $i;
        SEED=$1
        GPU=$2
        OUTPUT_DIR=../masks/global/$TASK_NAME/seed_$SEED/
        mkdir -p $OUTPUT_DIR
        export CUDA_VISIBLE_DEVICES=$GPU
        python find_global_weight_mask.py \
                --model_type bert \
                --model_name_or_path ../models/finetuned/$TASK_NAME/seed_$SEED/ \
                --task_name $TASK_NAME \
                --data_dir $GLUE_DIR/$TASK_NAME \
                --max_seq_length 128 \
                --output_dir $OUTPUT_DIR \
                --seed $SEED \
                --batch_size 64 &
    done
    wait;
done

# Compute Super Masks For HEAD+MLP Pruning
python compute_super_masks.py