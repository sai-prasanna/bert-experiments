# GLUE

Based on the script [`run_glue.py`](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py).


## Dataset preparation
Download GLUE data using `data/download_glue.py`. MRPC should be downloaded using the instructions provided in the `data/download_glue.py` docstring.

Data for all tasks should be downloadedin `data/glue` directory.

## Training 5 seeds on all tasks

Run 

``` sh
./train.sh
```

Five models for each tasks would be trained and placed in `finetuned_models/` directory.

## Evaluating on 5 seeds with different experiment settings

```sh
CUDA_VISIBLE_DEVICES=0 python franken_bert.py --data_dir ../data/glue/ --models_dir ../finetuned_models --do_lower_case --per_gpu_eval_batch_size 128 --model_type bert --output_dir ../experiments --experiment baseline
```

The `--experiment` parameter defines which set of experiments to run.

1. baseline
2. randomize_embeddings
3. randomize_qkv
4. randomize_qkv_together
5. zero_out_qkv
6. revert_embeddings
7. revert_qkv

The experiment results will be stored as json with the five seed mean and standard deviation.

### Baseline - Fine-tuned 5 seed mean results

| Task  | Metric                       | Result      |
|-------|------------------------------|-------------|
| CoLA  | Matthew's corr               | 56.03       |
| SST-2 | Accuracy                     | 92.56       |
| MRPC  | F1/Accuracy                  | 88.67/83.92 |
| STS-B | Person/Spearman corr.        | 88.65/88.31 |
| QQP   | Accuracy/F1                  | 90.91/87.82 |
| MNLI  | Matched acc./Mismatched acc. | 84.48/84.75 |
| QNLI  | Accuracy                     | 91.53       |
| RTE   | Accuracy                     | 63.24       |
| WNLI  | Accuracy                     | 33.80       |