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
python bert_base_fine_tune.py --data_dir ../../data/MNLI/ --output_dir checkpoints/ --do_train --do_eval --do_lower_case --num_train_epochs 3 --gpu_list 0 1 2 3
# generate HANS predictions - this will generate a file `hans_predictions.txt` inside the checkpoints/ directory
python test_hans.py --data_dir ../../data/hans --model_type bert --model_name_or_path checkpoints/ --do_eval --do_lower_case --max_seq_length 128 --output_dir checkpoints/ --task_name hans
# evaluate HANS predictions
python evaluate_heur_output.py --predictions checkpoints/hans_predictions.txt --evaluation_set ../../data/heuristics_evaluation_set.txt > ../../results/hans_results.txt
```

#### Pre-trained(frozen) + classifier on MNLI

```bash
cd src/mnli
# Train classifier on MNLI with frozen pre-trained layers
python bert_pretrained_mnli.py --data_dir ../../data/MNLI/ --output_dir pretrained_checkpoints/ --do_train --do_eval --do_lower_case --num_train_epochs 3 --gpu_list 0 1 2 3
# generate HANS predictions - this will generate a file `hans_predictions.txt` inside the checkpoints/ directory
python test_hans.py --data_dir ../../data/hans --model_type bert --model_name_or_path pretrained_checkpoints/ --do_eval --do_lower_case --max_seq_length 128 --output_dir pretrained_checkpoints/ --task_name hans
# evaluate HANS predictions
python evaluate_heur_output.py --predictions pretrained_checkpoints/hans_predictions.txt --evaluation_set ../../data/heuristics_evaluation_set.txt > ../../results/pretrained_hans_results.txt
```



### Models

| S.No | Model                         | Entailed results                                                      |  Non entailed results                                         |
|------|-------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------|
| 1    | [Bert base uncased fine-tuned](https://drive.google.com/file/d/1qv582bbpPVGoxnAr0vMOLsDwBiPXDOXp/view?usp=sharing)  | lexical_overlap: 0.9102 /  subsequence: 0.9256 /  constituent: 0.9508 | lexical_overlap: 1.0 /  subsequence: 1.0 /  constituent: 1.0  |
| 2    | [Bert base uncased pre-trained](https://drive.google.com/file/d/1hwFlMj5yjpEEp_Q0bRvRvaW61P8cXU8b/view?usp=sharing) | lexical_overlap: 1.0 /  subsequence: 1.0 / constituent: 1.0           | lexical_overlap: 0 .0 /  subsequence: 0.0 / constituent: 0.0  |