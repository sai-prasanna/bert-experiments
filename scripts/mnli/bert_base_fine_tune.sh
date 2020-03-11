#! /bin/bash
# Usage: ./bert_base_fine_tune.sh <data_dir> <output_dir> <num_gpus>

if [ ! $# -eq 3  ]; then  # check if only argument is passed
  echo "Invalid (or) insufficient parameters"
  echo "Usage: ./bert_base_fine_tune.sh <data_dir> <output_dir> <num_gpus>"
  exit 1
fi

script_dir=`dirname "$(readlink -f "$0")"`
src_dir="$script_dir/../../src/mnli"
data_dir=$1
output_dir=$2
world_size=$3

for i in $(seq 1 $world_size); do
    echo $i
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=$world_size RANK=$i python $src_dir/bert_base_fine_tune.py --data_dir ../../data/MNLI --output_dir ../../checkpoints --do_train --do_eval --do_lower_case --num_train_epochs 3 --local_rank $i &
done
