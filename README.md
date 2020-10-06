# When BERT Plays the Lottery, All Tickets Are Winning

Large Transformer-based models were shown to be reducible to a smaller number of self-attention heads and layers. We consider this phenomenon from the perspective of the lottery ticket hypothesis, using both structured and magnitude pruning. For fine-tuned BERT, we show that (a) it is possible to find subnetworks achieving performance that is comparable with that of the full model, and (b) similarly-sized subnetworks sampled from the rest of the model perform worse. Strikingly, with structured pruning even the worst possible subnetworks remain highly trainable, indicating that most pre-trained BERT weights are potentially useful.
We also study the "good" subnetworks to see if their success can be attributed to superior linguistic knowledge, but find them unstable, and not explained by meaningful self-attention patterns.

## Environment

Install the requirements in your python 3.7.7 virtual environment.

```sh
pip install -r requirements.txt
```

These experiments were done on multi-gpu environment, were some experiments, benchmarks were run parallel. So some changes to the bash scripts to make it work for your environment.

## Dataset

1. Download the GLUE dataset using `data/download_glue.py` and `data/download_mnli_data.py`. Follow the instructions in `data/download_glue.py` docstring for MRPC. 
2. All data for the tasks should be organized in `data/glue/task_name/` structure.
3. Extract the attention pattern classification labelled data.
    ```sh
    cd data
    tar -xvf head_classification_data.tar.gz
    ```

## Training, Masking, and Evaluation

Switch cwd to src (`cd src`) as many paths are relative from that directory. 

1. Fine-tune the BERT on GLUE tasks 
```sh
./train.sh
```
2. Obtain the masks
```sh
./find_masks.sh
```

3. Train models with the masks applied in good, random and bad settings.
```sh
./train_with_masks.sh
```

4. Evaluate the trained models
```sh
./evaluate.sh
```

Note: These experiments were run through course of time and now stiched together into single scripts. So it might be better to run
the training and evaluation commands in them one by one.

5. Train the CNN classifier on attention patterns normed and raw.
```sh
python classify_attention_patterns.py
python classify_normed_patterns.py
```
These only train the classifier.

## Evaluation Analysis and Final Results

These are primarily done in jupyter notebooks in `experiment_analysis` directory. There are many experimental notebooks there. Here are the important ones used to generate results included in the paper.

1. [Importance pruning Heatmaps](experiment_analysis/component_heatmaps.ipynb). Ignore the final "train_subset" and "hans" settings.
2. [Magnitude pruning Heatmap](experiment_analysis/global_magnitude_pruning_heatmaps.ipynb)
3. [Overlap of surviving components](experiment_analysis/common_components.ipynb)
4. [Generate the random baseline](experiment_analysis/frequency_baseline.ipynb)
5. [Attention Classification Patterns](experiment_analysis/head_classification.ipynb)
6. [Evaluation Result Comparisons and table](experiment_analysis/eval_performance_analysis.ipynb)
7. [Statistics on mask correlation across seeds](experiment_analysis/statistics_on_pruning.ipynb)
