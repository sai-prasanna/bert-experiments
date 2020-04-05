import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from hans_processors import HansProcessor
from hans_processors import hans_convert_examples_to_features as convert_examples_to_features
from transformers import (
    WEIGHTS_NAME,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

def create_eval_loader(data_dir, max_seq_length, model_type, model_name_or_path, tokenizer, overwrite_cache, eval_batch_size):
    eval_dataset, label_list = load_and_cache_examples(data_dir, max_seq_length, model_type, model_name_or_path, tokenizer, overwrite_cache)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
    return eval_dataloader, label_list

def load_and_cache_examples(data_dir, max_seq_length, model_type, model_name_or_path, tokenizer, overwrite_cache):
    processor = HansProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, model_name_or_path.split("/"))).pop(),
            str(max_seq_length),
            "hans",
        ),
    )

    label_list = processor.get_labels()

    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        if model_type in ["roberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            output_mode="classification",
            pad_on_left=bool(model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    all_pair_ids = torch.tensor([int(f.pairID) for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_pair_ids)
    return dataset, label_list

def predict(device, model, tokenizer, eval_dataloader, label_list):
    model = model.to(device)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if model.config.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if model.config.model_type in ["bert", "xlnet"] else None
                )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            pair_ids = batch[4].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            pair_ids = np.append(pair_ids, batch[4].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    
    return {f"ex{pid}": label_list[int(pred)] for pid, pred in zip(pair_ids, preds)}


def evaluate(predictions, hans_data_dir):

    def format_label(label):
        if label == "entailment":
            return "entailment"
        else:
            return "non-entailment"

    fi = open(hans_data_dir + '/heuristics_evaluation_set.txt', "r")

    correct_dict = {}
    first = True

    heuristic_list = []
    subcase_list = []
    template_list = []

    for line in fi:
        if first:
            labels = line.strip().split("\t")
            idIndex = labels.index("pairID")
            first = False
            continue
        else:
            parts = line.strip().split("\t")
            this_line_dict = {}
            for index, label in enumerate(labels):
                if label == "pairID":
                    continue
                else:
                    this_line_dict[label] = parts[index]
            correct_dict[parts[idIndex]] = this_line_dict

            if this_line_dict["heuristic"] not in heuristic_list:
                heuristic_list.append(this_line_dict["heuristic"])
            if this_line_dict["subcase"] not in subcase_list:
                subcase_list.append(this_line_dict["subcase"])
            if this_line_dict["template"] not in template_list:
                template_list.append(this_line_dict["template"])

    heuristic_ent_correct_count_dict = {}
    subcase_correct_count_dict = {}
    template_correct_count_dict = {}
    heuristic_ent_incorrect_count_dict = {}
    subcase_incorrect_count_dict = {}
    template_incorrect_count_dict = {}
    heuristic_nonent_correct_count_dict = {}
    heuristic_nonent_incorrect_count_dict = {}



    for heuristic in heuristic_list:
        heuristic_ent_correct_count_dict[heuristic] = 0
        heuristic_ent_incorrect_count_dict[heuristic] = 0
        heuristic_nonent_correct_count_dict[heuristic] = 0 
        heuristic_nonent_incorrect_count_dict[heuristic] = 0

    for subcase in subcase_list:
        subcase_correct_count_dict[subcase] = 0
        subcase_incorrect_count_dict[subcase] = 0

    for template in template_list:
        template_correct_count_dict[template] = 0
        template_incorrect_count_dict[template] = 0

    for key in correct_dict:
        traits = correct_dict[key]
        heur = traits["heuristic"]
        subcase = traits["subcase"]
        template = traits["template"]

        guess = format_label(predictions[key])
        correct = traits["gold_label"]

        if guess == correct:
            if correct == "entailment":
                heuristic_ent_correct_count_dict[heur] += 1
            else:
                heuristic_nonent_correct_count_dict[heur] += 1

            subcase_correct_count_dict[subcase] += 1
            template_correct_count_dict[template] += 1
        else:
            if correct == "entailment":
                heuristic_ent_incorrect_count_dict[heur] += 1
            else:
                heuristic_nonent_incorrect_count_dict[heur] += 1
            subcase_incorrect_count_dict[subcase] += 1
            template_incorrect_count_dict[template] += 1
    
    results = {
        "entailed": {},
        "non_entailed": {},
        "subcase": {},
        "template": {}
    }
    print("Heuristic entailed results:")
    entailed_correct = 0
    entailed_total = 0
    for heuristic in heuristic_list:
        correct = heuristic_ent_correct_count_dict[heuristic]
        incorrect = heuristic_ent_incorrect_count_dict[heuristic]
        total = correct + incorrect
        entailed_correct += correct
        entailed_total += total
        percent = correct * 1.0 / total
        results["entailed"][heuristic] = percent
        print(heuristic + ": " + str(percent))
    entailed_accuracy = entailed_correct * 1.0/entailed_total
    results["entailed_accuracy"] = entailed_accuracy

    print(f"\nEntailed accuracy: {entailed_accuracy}")

    non_entailed_correct = 0
    non_entailed_total = 0
    print("")
    print("Heuristic non-entailed results:")
    for heuristic in heuristic_list:
        correct = heuristic_nonent_correct_count_dict[heuristic]
        incorrect = heuristic_nonent_incorrect_count_dict[heuristic]
        total = correct + incorrect
        percent = correct * 1.0 / total
        non_entailed_correct += correct
        non_entailed_total += total
        results["non_entailed"][heuristic] = percent

        print(heuristic + ": " + str(percent))
    
    non_entailed_accuracy = non_entailed_correct * 1.0/non_entailed_total
    results["non_entailed_accuracy"] = non_entailed_accuracy
    print(f"\nNon-Entailed accuracy: {non_entailed_accuracy}")

    total_accuracy = (entailed_correct + non_entailed_correct)  * 1.0/(entailed_total + non_entailed_total)
    results["accuracy"] = total_accuracy
    print(f"\nOverall accuracy: {total_accuracy}")
    
    print("")
    print("Subcase results:")
    for subcase in subcase_list:
        correct = subcase_correct_count_dict[subcase]
        incorrect = subcase_incorrect_count_dict[subcase]
        total = correct + incorrect
        percent = correct * 1.0 / total
        results["subcase"][subcase] = percent

        print(subcase + ": " + str(percent))

    print("")
    # print("Template results:")
    for template in template_list:
        correct = template_correct_count_dict[template]
        incorrect = template_incorrect_count_dict[template]
        total = correct + incorrect
        percent = correct * 1.0 / total
        results["template"][template] = percent

        # print(template + ": " + str(percent))
    return results


def evaluate_mnli()

def build_parser():
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
        "--model_dir",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model.",
    )
    parser.add_argument(
        "--device_id",
        default=-1,
        type=int,
        required=True,
        help="GPU id or -1 for cpu",
    )
    parser.add_argument(
        "--max_seq_len",
        default=128,
        type=int,
        help="Max sequence length to be used for evaluation.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=128,
        type=int,
        help="Batch size for evaluation.",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    device = torch.device("cpu") if args.device_id == -1 else torch.device(f"cuda:{args.device_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    eval_dataloader, label_list = create_eval_loader(data_dir=args.data_dir, 
                                                 max_seq_length=args.max_seq_len, 
                                                 model_type=model.config.model_type, 
                                                 model_name_or_path=args.model_dir, 
                                                 tokenizer=tokenizer, 
                                                 overwrite_cache=False, 
                                                 eval_batch_size=args.eval_batch_size)
    
    
    predictions = predict(device, model, tokenizer, eval_dataloader, label_list)
    evaluate(predictions, args.data_dir)

if __name__ == "__main__":
    main()