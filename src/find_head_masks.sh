export GLUE_DIR=../data/glue
mkdir -p ../masks/

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
                --save_mask_all_iterations \
                --try_masking &
    done
    wait;
done
