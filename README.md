# bert-experiments

Steps to run the experiments:

1. Download MNLI data

```bash
cd data
chmod +x download_mnli_data.py
./download_mnli_data.py --data_dir MNLI/
```

2. Run fine-tuning

```bash
cd src/mnli
python bert_base_fine_tune.py --data_dir ../../data/MNLI/ --output_dir checkpoints/ --do_train --do_eval --do_lower_case --num_train_epochs 3 --gpu_list 0 1 2 3
```

3. Generate HANS predictions
```bash
cd src/mnli
python test_hans.py --data_dir ../../data/hans --model_type bert --model_name_or_path checkpoints/ --do_eval --do_lower_case --max_seq_length 128 --output_dir checkpoints/ --task_name hans
```
This will generate a file `hans_predictions.txt` inside the checkpoints/ directory

4. Evaluate HANS predictions
```bash
cd src/mnli
python evaluate_heur_output.py --predictions checkpoints/hans_predictions.txt --evaluation_set ../../data/heuristics_evaluation_set.txt > ../../results/hans_results.txt
```
