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
python bert_base_fine_tune.py --data-dir ../../data/MNLI/ --output-dir checkpoints/ --do_train --do_eval --do_lower_case --num_train_epochs 3 --gpu_list 0 1 2 3
```