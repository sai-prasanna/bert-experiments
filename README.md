# bert-experiments

Steps to run the experiments:

### Download MNLI data

```bash
cd data
chmod +x download_mnli_data.py
./download_mnli_data.py --data_dir MNLI/
```

### Evaluating BERT models on HANS


#### Fine-tuned on MNLI

```bash
cd src/mnli
# fine-tune on MNLI
CUDA_VISIBLE_DEVICES=3 python run_glue.py --data_dir ../../data/MNLI --output_dir checkpoints/finetuned/ --do_train --do_eval --do_lower_case --evaluate_during_training --per_gpu_eval_batch_size 128 --model_type bert --model_name_or_path bert-base-uncased --task_name mnli  --save_steps 5000 --logging_steps 5000 --train_mode finetune 

# Predict and evaluate on HANS
python python evaluate.py --model_dir checkpoints/finetuned/ --data_dir ../../data/hans/ --device_id 0
```

#### Pre-trained(frozen) + classifier on MNLI

```bash
cd src/mnli
# Train classifier on MNLI with frozen pre-trained layers
CUDA_VISIBLE_DEVICES=3 python run_glue.py --data_dir ../../data/MNLI --output_dir checkpoints/finetuned/ --do_train --do_eval --do_lower_case --evaluate_during_training --per_gpu_eval_batch_size 128 --train_mode finetune --model_type bert --model_name_or_path bert-base-uncased --task_name mnli  --save_steps 5000 --logging_steps 5000 --train_mode frozen

# Predict and evaluate on HANS
python python evaluate.py --model_dir checkpoints/frozen/ --data_dir ../../data/hans/ --device_id 0
```

#### Random BERT + classifier on MNLI

```bash
cd src/mnli
# Train classifier on MNLI with frozen pre-trained layers
CUDA_VISIBLE_DEVICES=3 python run_glue.py --data_dir ../../data/MNLI --output_dir checkpoints/finetuned/ --do_train --do_eval --do_lower_case --evaluate_during_training --per_gpu_eval_batch_size 128 --train_mode finetune --model_type bert --model_name_or_path bert-base-uncased --task_name mnli  --save_steps 5000 --logging_steps 5000 --train_mode random

# Predict and evaluate on HANS
python python evaluate.py --model_dir checkpoints/random/ --data_dir ../../data/hans/ --device_id 0
```

### Models

| S.No | Model                         | Entailed results                                                      |  Non entailed results                                         |
|------|-------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------|
| 1    | [Bert base uncased fine-tuned](https://drive.google.com/file/d/1qv582bbpPVGoxnAr0vMOLsDwBiPXDOXp/view?usp=sharing)  | lexical_overlap: 0.9102 /  subsequence: 0.9256 /  constituent: 0.9508 | lexical_overlap: 0.1948 /  subsequence: 0.1156 /  constituent: 0.081 |
| 2    | [Bert base uncased pre-trained](https://drive.google.com/file/d/1hwFlMj5yjpEEp_Q0bRvRvaW61P8cXU8b/view?usp=sharing) | lexical_overlap: 1.0 /  subsequence: 1.0 / constituent: 1.0           | lexical_overlap: 0 .0 /  subsequence: 0.0 / constituent: 0.0  |
