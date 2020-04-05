## GLUE

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py).

Fine-tuning the library models for sequence classification on the GLUE benchmark: [General Language Understanding
Evaluation](https://gluebenchmark.com/). This script can fine-tune the following models: BERT, XLM, XLNet and RoBERTa.

GLUE is made up of a total of 9 different tasks. We get the following results on the dev set of the benchmark with an
uncased  BERT base model (the checkpoint `bert-base-uncased`). All experiments ran single V100 GPUs with a total train
batch sizes between 16 and 64. Some of these tasks have a small dataset and training can lead to high variance in the results
between different runs. We report the median on 5 runs (with different seeds) for each of the metrics.

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthew's corr               | 49.23       |
| SST-2 | Accuracy                     | 91.97       |
| MRPC  | F1/Accuracy                  | 89.47/85.29 |
| STS-B | Person/Spearman corr.        | 83.95/83.70 |
| QQP   | Accuracy/F1                  | 88.40/84.31 |
| MNLI  | Matched acc./Mismatched acc. | 80.61/81.08 |
| QNLI  | Accuracy                     | 87.46       |
| RTE   | Accuracy                     | 61.73       |
| WNLI  | Accuracy                     | 45.07       |

Some of these results are significantly different from the ones reported on the test set
of GLUE benchmark on the website. For QQP and WNLI, please refer to [FAQ #12](https://gluebenchmark.com/faq) on the webite.

Before running any one of these GLUE tasks you should download the
[GLUE data](https://gluebenchmark.com/tasks) by running
[this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
and unpack it to some directory `$GLUE_DIR`.

```bash
export GLUE_DIR=../data/glue
export TASK_NAME=MRPC

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
  --output_dir ../experiments/$TASK_NAME/ \
  --seed 13
```

where task name can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, WNLI.

The dev set results will be present within the text file `eval_results.txt` in the specified output_dir.
In case of MNLI, since there are two separate dev sets (matched and mismatched), there will be a separate
output folder called `/tmp/MNLI-MM/` in addition to `/tmp/MNLI/`.

The code has not been tested with half-precision training with apex on any GLUE task apart from MRPC, MNLI,
CoLA, SST-2. The following section provides details on how to run half-precision training with MRPC. With that being
said, there shouldn’t be any issues in running half-precision training with the remaining GLUE tasks as well,
since the data processor for each task inherits from the base class DataProcessor.
