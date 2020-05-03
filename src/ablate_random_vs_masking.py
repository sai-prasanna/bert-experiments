# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random
import pathlib
import copy
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import scipy.stats

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    FlaubertConfig,
    FlaubertForSequenceClassification,
    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
#from transformers import glue_compute_metrics as compute_metrics
from glue_metrics import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from experiment_impact_tracker.compute_tracker import ImpactTracker

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate(args, task_name, data_dir,  model, tokenizer, head_mask=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if task_name == "mnli" else (task_name,)
    results = {}
    for eval_task in eval_task_names:
        eval_dataset = load_and_cache_examples(args, data_dir, eval_task, tokenizer, evaluate=True)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        if head_mask is not None:
            head_mask = torch.tensor(head_mask, device=args.device)
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs, head_mask=head_mask)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        processor = processors[eval_task]()
        label_list = processor.get_labels()
        if eval_task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        label_map = {int(i): label for i, label in enumerate(label_list)}

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
            pred_outputs = [label_map[p] for p in preds]
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
            pred_outputs = [float(p) for p in preds]
        result = compute_metrics(eval_task, preds, out_label_ids)
        
        for k, v in result.items():
            if "mnli" in eval_task:
                k = f"{eval_task}_{k}"
            results[k] = v
        
        if "predictions" not in results:
            results["predictions"] = {}
        if "mnli" in eval_task:
            results["predictions"][eval_task] = pred_outputs
        else:
            results["predictions"] = pred_outputs
    return results


def load_and_cache_examples(args, data_dir, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            # list(filter(None, args.model_name_or_path.split("/"))).pop(), Single cache for each task
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the benchmarks would be put.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="none",
        required=True,
        help="The randomization experiment to run.")
    parser.add_argument(
        "--include_predictions", action="store_true", help="Set this flag if you want to save the predictions for the experiment.",
    )
    parser.add_argument(
        "--models_dir",
        default=None,
        type=str,
        required=True,
        help="The fine-tuned models directory where all the tasks with respective model seed checkpoints are stored.",
    )
    parser.add_argument(
        "--masks_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where final masks after pruning are stored.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    # Other parameters
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()


    args.local_rank = -1
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device
    args.model_type = args.model_type.lower()
    args.experiment = args.experiment.lower()
    args.output_dir = f"{args.output_dir}/{args.experiment}"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    tracker = ImpactTracker(args.output_dir)
    tracker.launch_impact_monitor()

    # Prepare GLUE task
    results = evaluate_all_tasks(args)
    results_file_path = f"{args.output_dir}/results.json"
    write_results(results, results_file_path)


def evaluate_all_tasks(args):
    # Prepare GLUE task
    tasks = ["CoLA", "MNLI", "MRPC", "QNLI", "QQP", "RTE", "SST-2", "STS-B", "WNLI"]
    all_task_results = {}
    for task in tasks:
        metrics = evaluate_task_with_initialization(args, task)
        all_task_results[task] = metrics
    return all_task_results

def evaluate_task_with_initialization(args, task: str):
    models_dir = pathlib.Path(args.models_dir)
    task_dir = models_dir / task
    task_name = task.lower()
    
    processor = processors[task_name]()
    args.output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    seed_results = []

    masks_path = pathlib.Path(args.masks_dir)

    for seed_dir in task_dir.glob("seed_*"):
        head_mask = np.load(masks_path / task / seed_dir.stem / "head_mask.npy")

        args.model_name_or_path = str(seed_dir)
        config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        

        # masked Evaluation
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        set_seed(args)
        model.to(args.device)
        data_dir = f"{args.data_dir}/{task}"
        if args.experiment == "head_mask_apply":
            # Set task specific args
            
            result = evaluate(args, task_name, data_dir, model, tokenizer, head_mask=head_mask)
        elif args.experiment == "head_mask_randomize":
            pattern = r"layer.(\d+).attention.(output.dense|self.key|self.query|self.value)"
            for name, module in model.named_modules():
                match = re.search(pattern, name)
                if match:
                    layer = int(match.group(1))
                    head_size = model.config.hidden_size // model.config.num_attention_heads
                    for head in range(model.config.num_attention_heads):
                        if head_mask[layer][head] == 0:
                            start = head * head_size
                            end = (head + 1) * head_size
                            module.weight.data[start:end].normal_(mean=0.0, std=model.config.initializer_range)
                            logger.info(f"Randomizing {name} in head {layer}")
            result = evaluate(args, task_name, data_dir, model, tokenizer)
        else:
            raise RuntimeError("Invalid experiment")
        del result["predictions"]
        seed_results.append(result)

    task_result = {}
    for key in seed_results[0]:
        task_result[key] = scipy.stats.norm.fit([result[key] for result in seed_results])
    return task_result

def write_results(results, output_file_path):
    with open(output_file_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
