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
import torch.nn.utils.prune as prune
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
from run_glue import processors, output_modes, add_masks, load_trained_model
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

from model_bert import BertForSequenceClassification
from config_bert import BertConfig

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

def evaluate(args, task_name, data_dir,  model, tokenizer):
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
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
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
        default="baseline",
        required=True,
        help="The randomization experiment to run. Default `baseline` does no randomization")
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
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type",
    )
    # Other parameters
    parser.add_argument(
        "--global_masks_dir",
        default=None,
        type=str,
        required=False,
        help="Global masks to be applied before running the experiment. (Used only for baseline experiment)",
    )
    parser.add_argument(
        "--global_mask_file_name",
        default=None,
        type=str,
        required=False,
        help="Global masks to be applied before running the experiment. (Used only for baseline experiment)",
    )
    parser.add_argument(
        "--head_masks_dir",
        default=None,
        type=str,
        required=False,
        help="Head masks to be applied before running the experiment. (Used only for baseline experiment)",
    )
    parser.add_argument(
        "--mlp_masks_dir",
        default=None,
        type=str,
        required=False,
        help="MLP masks to be applied before running the experiment. (Used only for baseline experiment)",
    )
    parser.add_argument(
        "--mask_mode",
        choices=["use", "invert", "random", "bad"], 
        default="use",
        help="use,invert,random"
    )
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
        "--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.",
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
    experiment_map = {
        # Only the `baseline` is used for the paper results.

        "baseline": experiment_baseline,
        # The rest can be ignored.
        "randomize_embeddings": experiment_randomize_embeddings,
        "randomize_qkv": experiment_randomize_qkv,
        "randomize_fc": experiment_randomize_fc,

        "randomize_qkv_together": experiment_randomize_qkv_together,
        "randomize_qkv_together_pairwise": experiment_randomize_qkv_together_pairwise,
        "zero_out_qkv": experiment_zero_out_qkv,
        
        "randomize_full_layerwise": experiment_randomize_full_layerwise,
        "randomize_components": experiment_randomize_components,
        "revert_embeddings": experiment_revert_embeddings,
        "revert_qkv": experiment_revert_qkv,
        "revert_fc": experiment_revert_fc,
        "revert_embeddings_rotate": experiment_revert_embeddings_rotate,

        "ablate_residuals": experiment_ablate_residuals,

        "ablate_pruning": experiment_prune
    }
    experiment_map[args.experiment](args)


def experiment_baseline(args):
    results = evaluate_all_tasks_with_initialization(args, lambda x: x)
    results_file_path = f"{args.output_dir}/results.json"
    write_results(results, results_file_path)


def experiment_randomize_components(args):
    def get_randomizer(component_pattern):
        def randomization_func(model):
            for name, module in model.named_modules():
                if re.search(component_pattern, name):
                    logger.info(f"\nMatched - {name}\n")
                    module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            return model
        return randomization_func

    results = evaluate_all_tasks_with_initialization(args, lambda x: x)
    results_file_path = f"{args.output_dir}/baseline.json"
    write_results(results, results_file_path)

    results = evaluate_all_tasks_with_initialization(args, get_randomizer(r"word_embedding"))
    results_file_path = f"{args.output_dir}/randomize_embeddings.json"
    write_results(results, results_file_path)
    
    pattern = r"layer.\d+.(attention.self.value|attention.self.query|attention.self.key)"

    results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
    results_file_path = f"{args.output_dir}/randomize_qkv.json"
    write_results(results, results_file_path)
    
    pattern = r"layer.\d+.(attention.output.dense|intermediate.dense|output.dense)"
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
    results_file_path = f"{args.output_dir}/randomize_all.json"
    write_results(results, results_file_path)


class BertOutputWithoutResiduals(nn.Module):
    def __init__(self, dense, layer_norm, dropout_layer):
        super().__init__()
        self.dense = dense
        self.LayerNorm = layer_norm
        self.dropout = dropout_layer

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + hidden_states)
        return hidden_states

def experiment_ablate_residuals(args):
    def get_randomizer(residual_remove_layers, component_pattern):
        def randomization_func(model):
            for layer_id in residual_remove_layers:
                output_layer = model.bert.encoder.layer[layer_id].output
                output_wo_residuals = BertOutputWithoutResiduals(output_layer.dense, output_layer.LayerNorm, output_layer.dropout)
                model.bert.encoder.layer[layer_id].output = output_wo_residuals
                logger.info(f"Removed Residuals in {layer_id}\n")
            logger.info(f"Modified model - \n {model} \n")
            for name, module in model.named_modules():
                if component_pattern is not None and re.search(component_pattern, name):
                    logger.info(f"\nMatched - {name}\n")
                    module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            return model
        return randomization_func

    results = evaluate_all_tasks_with_initialization(args,  get_randomizer(list(range(12)), None))
    results_file_path = f"{args.output_dir}/baseline_remove_res_layer_all.json"
    write_results(results, results_file_path)

    for layer in range(12):
        results = evaluate_all_tasks_with_initialization(args,  get_randomizer([layer], None))
        results_file_path = f"{args.output_dir}/baseline_remove_res_layer_{layer}.json"
        write_results(results, results_file_path)


def experiment_randomize_embeddings(args):
    def randomize_embeddings(model):
        for name, module in model.named_modules():
            if "word_embedding" in name:
                module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        return model

    results = evaluate_all_tasks_with_initialization(args, randomize_embeddings)
    results_file_path = f"{args.output_dir}/results.json"
    write_results(results, results_file_path)

def experiment_revert_embeddings(args):
    _, model_class, _ = MODEL_CLASSES[args.model_type]
    
    if args.model_type != "bert":
        raise NotImplementedError("Logic for non bert models are not implemented for this experiment.")
    
    if args.do_lower_case:
        orignal_model = model_class.from_pretrained("bert-base-uncased")
    else:
        orignal_model = model_class.from_pretrained("bert-base-cased")
    orignal_model = orignal_model.eval()
    original_model_map = {name: module for name, module in orignal_model.named_modules()}

    def revert_embeddings(model):
        for name, module in model.named_modules():
            if "word_embedding" in name:
                module.weight.data = copy.deepcopy(original_model_map[name].weight.data)
        return model

    results = evaluate_all_tasks_with_initialization(args, revert_embeddings)
    results_file_path = f"{args.output_dir}/revert_embeddings_results.json"
    write_results(results, results_file_path)

def experiment_revert_embeddings_rotate(args):
    _, model_class, _ = MODEL_CLASSES[args.model_type]
    
    if args.model_type != "bert":
        raise NotImplementedError("Logic for non bert models are not implemented for this experiment.")
    
    if args.do_lower_case:
        orignal_model = model_class.from_pretrained("bert-base-uncased")
    else:
        orignal_model = model_class.from_pretrained("bert-base-cased")
    orignal_model = orignal_model.eval()
    original_model_map = {name: module for name, module in orignal_model.named_modules()}

    def revert_embeddings(model):
        for name, module in model.named_modules():
            if "word_embedding" in name:
                module.weight.data[0] = copy.deepcopy(original_model_map[name].weight.data[-1])
                module.weight.data[1:] = copy.deepcopy(original_model_map[name].weight.data[:-1])
        return model

    results = evaluate_all_tasks_with_initialization(args, revert_embeddings)
    results_file_path = f"{args.output_dir}/revert_embeddings_results.json"
    write_results(results, results_file_path)

def experiment_randomize_qkv(args):
    def get_randomizer(component):
        def randomization_func(model):
            for name, module in model.named_modules():
                if component in name:
                    module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            return model
        return randomization_func

    for component in tqdm(["query", "key", "value"]):
        component_name = f"attention.self.{component}"
        results = evaluate_all_tasks_with_initialization(args, get_randomizer(component_name))
        results_file_path = f"{args.output_dir}/{component}_layer_all_results.json"
        write_results(results, results_file_path)

        for layer in tqdm(range(12)):
            component_name = f"layer.{layer}.attention.self.{component}"
            results = evaluate_all_tasks_with_initialization(args, get_randomizer(component_name))
            results_file_path = f"{args.output_dir}/{component}_layer_{layer}_results.json"
            write_results(results, results_file_path)
    
def experiment_randomize_qkv_together(args):
    def get_randomizer(components):
        def randomization_func(model):
            for name, module in model.named_modules():
                if any((component in name for component in components)):
                    module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            return model
        return randomization_func
    components = ["attention.self.query", "attention.self.key", "attention.self.value"]
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(components))
    results_file_path = f"{args.output_dir}/qkv_layer_all_results.json"
    write_results(results, results_file_path)

    for layer in range(12):
        layer_components = [f"layer.{layer}.{component}" for component in components]
        results = evaluate_all_tasks_with_initialization(args, get_randomizer(layer_components))
        results_file_path = f"{args.output_dir}/qkv_layer_{layer}_results.json"
        write_results(results, results_file_path)

def experiment_randomize_qkv_together_pairwise(args):
    def get_randomizer(components):
        def randomization_func(model):
            for name, module in model.named_modules():
                if any((component in name for component in components)):
                    module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            return model
        return randomization_func
    components = ["attention.self.query", "attention.self.key", "attention.self.value"]
    for layer_1, layer_2 in zip(range(0, 11), range(1,12)):
        layer_components = [f"layer.{layer_1}.{component}" for component in components]
        layer_components.extend([f"layer.{layer_2}.{component}" for component in components])
        results = evaluate_all_tasks_with_initialization(args, get_randomizer(layer_components))
        results_file_path = f"{args.output_dir}/qkv_layers_{layer_1}_{layer_2}_results.json"
        write_results(results, results_file_path)


def experiment_zero_out_qkv(args):
    def get_randomizer(components):
        def randomization_func(model):
            for name, module in model.named_modules():
                if any((component in name for component in components)):
                    module.weight.data.zero_()
            return model
        return randomization_func
    
    for component in ["query", "key", "value"]:
        components = [f"attention.self.{component}"]
        results = evaluate_all_tasks_with_initialization(args, get_randomizer(components))
        results_file_path = f"{args.output_dir}/{component}_layer_all_results.json"
        write_results(results, results_file_path)

    components = [f"attention.self.query", f"attention.self.key"]
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(components))
    results_file_path = f"{args.output_dir}/qk_layer_all_results.json"
    write_results(results, results_file_path)

    components = [f"attention.self.query", f"attention.self.key", "attention.self.value"]
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(components))
    results_file_path = f"{args.output_dir}/qkv_layer_all_results.json"
    write_results(results, results_file_path)

def experiment_prune(args):
    def get_pruner(p, random=False):
        def pruning_func(model):
            parameters_to_prune = []
            for layer in model.bert.encoder.layer:
                parameters = [
                    (layer.attention.self.key, 'weight'),
                    (layer.attention.self.key, 'bias'),
                    (layer.attention.self.query, 'weight'),
                    (layer.attention.self.query, 'bias'),
                    (layer.attention.self.value, 'weight'),
                    (layer.attention.self.value, 'bias'),
                    (layer.attention.output.dense, 'weight'),
                    (layer.attention.output.dense, 'bias'),
                    (layer.intermediate.dense, 'weight'),
                    (layer.intermediate.dense, 'bias'),
                    (layer.output.dense, 'weight'),
                    (layer.output.dense, 'bias'),
                ]
                parameters_to_prune.extend(parameters)
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured if not random else prune.RandomUnstructured,
                amount=p,
            )
            return model
        return pruning_func
    
    for p in [0.5, 0.2, 0.7]:
        for random in [False, True]:
            results = evaluate_all_tasks_with_initialization(args, get_pruner(p, random))
            results_file_path = f"{args.output_dir}/p_{p}_{'random' if random else 'magnitude'}.json"
            write_results(results, results_file_path)



def experiment_randomize_fc(args):
    def get_randomizer(component_pattern):
        def randomization_func(model):
            for name, module in model.named_modules():
                if re.search(component_pattern, name):
                    logger.info(f"\nMatched - {name}\n")
                    module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
            return model
        return randomization_func

    pattern = r"layer.\d+.attention.output.dense|layer.\d+.intermediate.dense|layer.\d+.output.dense"
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
    results_file_path = f"{args.output_dir}/fc_a_i_o_layer_all_results.json"
    write_results(results, results_file_path)

    pattern = r"layer.\d+.attention.output.dense"
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
    results_file_path = f"{args.output_dir}/fc_a_all_results.json"
    write_results(results, results_file_path)

    pattern = r"layer.\d+.intermediate.dense"
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
    results_file_path = f"{args.output_dir}/fc_i_all_results.json"
    write_results(results, results_file_path)

    pattern = r"layer.\d+.output.dense"
    results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
    results_file_path = f"{args.output_dir}/fc_o_all_results.json"
    write_results(results, results_file_path)

    for layer in range(12):
        pattern = fr"layer.{layer}.(attention.output.dense|intermediate.dense|output.dense)"
        results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
        results_file_path = f"{args.output_dir}/fc_a_i_o_layer_{layer}_results.json"
        write_results(results, results_file_path)


def experiment_randomize_full_layerwise(args):
    def get_randomizer(component_pattern):
        def randomization_func(model):
            for name, module in model.named_modules():
                if re.search(component_pattern, name):
                    logger.info(f"\nMatched - {name}\n")
                    module.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
                    module.bias.data.zero_()
            return model
        return randomization_func
    
    for layer_1, layer_2 in zip(range(0, 11), range(1,12)):
        pattern = fr"layer.({layer_1}|{layer_2}).(attention.self.value|attention.self.query|attention.self.key|attention.output.dense|intermediate.dense|output.dense)"
        results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
        results_file_path = f"{args.output_dir}/randomize_layers_{layer_1}_{layer_2}_results.json"
        write_results(results, results_file_path)

    for layer in range(12):
        pattern = fr"layer.{layer}.(attention.self.value|attention.self.query|attention.self.key|attention.output.dense|intermediate.dense|output.dense)"
        results = evaluate_all_tasks_with_initialization(args, get_randomizer(pattern))
        results_file_path = f"{args.output_dir}/randomize_layer_{layer}_results.json"
        write_results(results, results_file_path)
    


def experiment_revert_qkv(args):
    _, model_class, _ = MODEL_CLASSES[args.model_type]
    
    if args.model_type != "bert":
        raise NotImplementedError("Logic for non bert models are not implemented for this experiment.")
    
    if args.do_lower_case:
        orignal_model = model_class.from_pretrained("bert-base-uncased")
    else:
        orignal_model = model_class.from_pretrained("bert-base-cased")
    orignal_model = orignal_model.eval()
    original_model_map = {name: module for name, module in orignal_model.named_modules()}

    def get_reverter(components):
        def revert_func(model):
            for name, module in model.named_modules():
                if any((component in name for component in components)):
                    module.weight.data = copy.deepcopy(original_model_map[name].weight.data)
                    module.bias.data = copy.deepcopy(original_model_map[name].bias.data)
            return model
        return revert_func
    
    for component in ["query", "key", "value"]:
        components = [f"attention.self.{component}"]
        results = evaluate_all_tasks_with_initialization(args, get_reverter(components))
        results_file_path = f"{args.output_dir}/{component}_layer_all_results.json"
        write_results(results, results_file_path)

    components = ["attention.self.query", "attention.self.key"]
    results = evaluate_all_tasks_with_initialization(args, get_reverter(components))
    results_file_path = f"{args.output_dir}/qk_layer_all_results.json"
    write_results(results, results_file_path)

    components = ["attention.self.query", "attention.self.value"]
    results = evaluate_all_tasks_with_initialization(args, get_reverter(components))
    results_file_path = f"{args.output_dir}/qv_layer_all_results.json"
    write_results(results, results_file_path)

    components = ["attention.self.value", "attention.self.key"]
    results = evaluate_all_tasks_with_initialization(args, get_reverter(components))
    results_file_path = f"{args.output_dir}/vk_layer_all_results.json"
    write_results(results, results_file_path)

    components = ["attention.self.query", "attention.self.key", "attention.self.value"]
    results = evaluate_all_tasks_with_initialization(args, get_reverter(components))
    results_file_path = f"{args.output_dir}/qkv_layer_all_results.json"
    write_results(results, results_file_path)


def experiment_revert_fc(args):
    _, model_class, _ = MODEL_CLASSES[args.model_type]
    
    if args.model_type != "bert":
        raise NotImplementedError("Logic for non bert models are not implemented for this experiment.")
    
    if args.do_lower_case:
        orignal_model = model_class.from_pretrained("bert-base-uncased")
    else:
        orignal_model = model_class.from_pretrained("bert-base-cased")
    orignal_model = orignal_model.eval()
    original_model_map = {name: module for name, module in orignal_model.named_modules()}

    def get_reverter(component_pattern):
        def init_func(model):
            for name, module in model.named_modules():
                if re.search(component_pattern, name):
                    module.weight.data = copy.deepcopy(original_model_map[name].weight.data)
                    module.bias.data = copy.deepcopy(original_model_map[name].bias.data)
            return model
        return init_func

    pattern = r"layer.\d+.attention.output.dense|layer.\d+.intermediate.dense|layer.\d+.output.dense"
    results = evaluate_all_tasks_with_initialization(args, get_reverter(pattern))
    results_file_path = f"{args.output_dir}/fc_a_i_o_layer_all_results.json"
    write_results(results, results_file_path)

    pattern = r"layer.\d+.attention.output.dense"
    results = evaluate_all_tasks_with_initialization(args, get_reverter(pattern))
    results_file_path = f"{args.output_dir}/fc_a_all_results.json"
    write_results(results, results_file_path)

    pattern = r"layer.\d+.intermediate.dense"
    results = evaluate_all_tasks_with_initialization(args, get_reverter(pattern))
    results_file_path = f"{args.output_dir}/fc_i_all_results.json"
    write_results(results, results_file_path)

    pattern = r"layer.\d+.output.dense"
    results = evaluate_all_tasks_with_initialization(args, get_reverter(pattern))
    results_file_path = f"{args.output_dir}/fc_o_all_results.json"
    write_results(results, results_file_path)



def evaluate_all_tasks_with_initialization(args, initialization_func):
    # Prepare GLUE task
    models_dir = pathlib.Path(args.models_dir)
    tasks = [p.stem for p in models_dir.iterdir()]
    all_task_results = {}
    for task in tasks:
        metrics, predictions = evaluate_task_with_initialization(args, task, initialization_func)
        all_task_results[task] = metrics
        if args.include_predictions:
            all_task_results[task]["predictions"] = predictions
    return all_task_results


def evaluate_task_with_initialization(args, task: str, initialization_func):
    models_dir = pathlib.Path(args.models_dir)
    task_dir = models_dir / task
    task_name = task.lower()
    
    processor = processors[task_name]()
    args.output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    seed_results = []
    seed_predictions = {}
    for seed_dir in task_dir.glob("seed_*"):
        args.model_name_or_path = str(seed_dir)
        # config = config_class.from_pretrained(
        #     args.model_name_or_path,
        #     num_labels=num_labels,
        #     finetuning_task=task_name,
        #     cache_dir=args.cache_dir if args.cache_dir else None,
        # )
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = load_trained_model(args.model_name_or_path, model_class, config_class)
        if args.global_masks_dir is not None:
            mask_file = pathlib.Path(args.global_masks_dir) / task / seed_dir.stem / args.global_mask_file_name
            masks = torch.load(mask_file)
            add_masks(model, masks)

        if args.head_masks_dir is not None:
            head_mask_file = pathlib.Path(args.head_masks_dir) / task / seed_dir.stem / "head_mask.npy"
            head_mask = np.load(head_mask_file)
            if args.mask_mode == "random":
                logger.info(f"Creating random head_mask with about {head_mask.sum()} elements")
                p_unpruned = head_mask.sum() / head_mask.size
                head_mask = np.zeros_like(head_mask)
                uniform_random = np.random.rand(*head_mask.shape)
                head_mask[uniform_random < p_unpruned] = 1
                logger.info(f"Random head_mask {head_mask} with {head_mask.sum()} elements")
            elif args.mask_mode == "invert":
                head_mask = 1 - head_mask
                logger.info(f"Invert head_mask {head_mask} with {head_mask.sum()} elements")
            elif args.mask_mode == "bad":
                total_good = int(head_mask.sum())
                total_bad = int((1-head_mask).sum())
                if total_good > total_bad:
                    bad_indices = np.argwhere(head_mask == 0).tolist()
                    remaining_indices = random.sample(np.argwhere(head_mask == 1).tolist(), total_good - total_bad)
                    bad_indices.extend(remaining_indices) # Remaining heads sampled from "good" heads.
                else:
                    bad_indices = random.sample(np.argwhere(head_mask == 0).tolist(), total_good)
                head_mask = np.zeros_like(head_mask)
                for idx in bad_indices:
                    head_mask[idx[0], idx[1]] = 1
                assert int(head_mask.sum()) == total_good
            head_mask = torch.from_numpy(head_mask)
            heads_to_prune = {}
            for layer in range(len(head_mask)):
                heads_to_mask = [h[0] for h in (1 - head_mask[layer].long()).nonzero().tolist()]
                heads_to_prune[layer] = heads_to_mask
            assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
            logger.info(f"Pruning heads {heads_to_prune}")
            model.prune_heads(heads_to_prune)
        if args.mlp_masks_dir is not None:
            mlp_mask_file = pathlib.Path(args.mlp_masks_dir) / task / seed_dir.stem / "mlp_mask.npy"
            mlp_mask = np.load(mlp_mask_file)
            if args.mask_mode == "random":
                p_unpruned = mlp_mask.sum() / mlp_mask.size
                mlp_mask = np.zeros_like(mlp_mask)
                uniform_random = np.random.rand(*mlp_mask.shape)
                mlp_mask[uniform_random < p_unpruned] = 1
            elif args.mask_mode == "invert":
                mlp_mask = 1 - mlp_mask
            elif args.mask_mode == "bad":
                total_good = int(mlp_mask.sum())
                total_bad = int((1-mlp_mask).sum())
                if total_good > total_bad:
                    bad_indices = np.argwhere(mlp_mask == 0).tolist()
                    remaining_indices = random.sample(np.argwhere(mlp_mask == 1).tolist(), total_good - total_bad)
                    bad_indices.extend(remaining_indices) # Remaining heads sampled from "good" heads.
                else:
                    bad_indices = random.sample(np.argwhere(mlp_mask == 0).tolist(), total_good)
                mlp_mask = np.zeros_like(mlp_mask)
                for idx in bad_indices:
                    mlp_mask[idx[0]] = 1
                assert int(mlp_mask.sum()) == total_good
            mlps_to_prune = [h[0] for h in (1 - torch.from_numpy(mlp_mask).long()).nonzero().tolist()]
            logger.info(f"MLPS to prune - {mlps_to_prune}")
            model.prune_mlps(mlps_to_prune)

        set_seed(args)
        model = initialization_func(model)

        model.to(args.device)            
        # Set task specific args
        data_dir = f"{args.data_dir}/{task}"
        
        result = evaluate(args, task_name, data_dir, model, tokenizer)
        seed_predictions[seed_dir.stem] = result["predictions"]
        del result["predictions"]
        logger.info(f"{args.experiment}: {task}: {seed_dir.stem}: {result}")
        seed_results.append(result)
    task_result = {}
    for key in seed_results[0]:
        task_result[key] = scipy.stats.norm.fit([result[key] for result in seed_results])
    return task_result, seed_predictions

def write_results(results, output_file_path):
    with open(output_file_path, "w") as fp:
        json.dump(results, fp, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()
