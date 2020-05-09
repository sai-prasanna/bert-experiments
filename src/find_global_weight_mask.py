import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.nn.utils.prune as prune
from tqdm import tqdm
from experiment_impact_tracker.compute_tracker import ImpactTracker


from run_glue import ALL_MODELS, MODEL_CLASSES, load_and_cache_examples, set_seed, output_modes, processors, parameters_to_prune
from glue_metrics import glue_compute_metrics as compute_metrics
from model_bert import BertForSequenceClassification
from config_bert import BertConfig


logger = logging.getLogger(__name__)
logging.getLogger("experiment_impact_tracker.compute_tracker.ImpactTracker").disabled = True



class L1UnstructuredInvert(prune.BasePruningMethod):
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones
    with the highest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount):
        # Check range of validity of pruning amount
        prune._validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the slargest k
            topk = torch.topk(
                torch.abs(t).view(-1), k=nparams_toprune, largest=True
            )
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(L1UnstructuredInvert, cls).apply(module, name, amount=amount)


def prune_model(model, amount, mode):

    params = parameters_to_prune(model)
    if mode == prune.Identity:
        prune.global_unstructured(
            params,
            pruning_method=mode,
        )
    else:
        prune.global_unstructured(
            params,
            pruning_method=mode,
            amount=amount,
        )
       
    return {k: v.cpu() for k, v in model.named_buffers()}


def evaluate(
    args, model, eval_dataloader
):
    """ This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    # Prepare our tensors
    model.eval()
    preds = None
    labels = None
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            outputs = model(
                input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids
            )
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  
        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_ids.detach().cpu().numpy(), axis=0)

    return preds, labels


def mask_global(args, model, eval_dataloader):
    """ This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    preds, labels = evaluate(args, model, eval_dataloader)
    preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
    original_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
    logger.info("Pruning: original score: %f, threshold: %f", original_score, original_score * args.masking_threshold)
    
    logger.info("Finding Gobal magnitude mask")

    model = model.to(torch.device("cpu"))
    new_mask = prune_model(model, amount=None, mode=prune.Identity)
    current_score = original_score
    

    total_steps = 10
    while total_steps != 0 and current_score >= original_score * args.masking_threshold:
        print("Current Score:", current_score)
        print("Original Score:", original_score)
        total_steps -= 1
        # Head New mask
        mask = {k:v.clone() for k, v in new_mask.items()}  # save current head mask
        print("Total pruned Amount", (sum([(1-v).sum() for v in mask.values()])/sum([v.numel() for v in mask.values()])))

        model = model.to(torch.device("cpu"))
        new_mask = prune_model(model, amount=args.masking_amount, mode=prune.L1Unstructured)

        model = model.to(args.device)
        preds, labels = evaluate(args, model, eval_dataloader)
        preds = np.argmax(preds, axis=1) if args.output_mode == "classification" else np.squeeze(preds)
        current_score = compute_metrics(args.task_name, preds, labels)[args.metric_name]
        logger.info(
            "Masking: current score: %f",
            current_score,
        )
    
    total_masked = sum([(1-v).sum() for v in mask.values()])
    total_elements = sum([v.numel() for v in mask.values()])
    logger.info(f"Prunned {total_masked/total_elements * 100:.2f}% of weights")
    torch.save(mask, os.path.join(args.output_dir, "magnitude_mask.p"))

    return mask, float(total_masked/total_elements)


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
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--data_subset", type=int, default=-1, help="If > 0: limit the data to a subset of data_subset instances."
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Whether to overwrite data in output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--dont_normalize_importance_by_layer", action="store_true", help="Don't normalize importance score by layers"
    )
    parser.add_argument(
        "--dont_normalize_global_importance",
        action="store_true",
        help="Don't normalize all importance scores between 0 and 1",
    )
    parser.add_argument(
        "--use_train_data", action="store_true", help="Use training set for computing masks"
    )
    parser.add_argument(
        "--masking_threshold",
        default=0.9,
        type=float,
        help="masking threshold in term of metrics (stop masking when metric < threshold * original metric value).",
    )
    parser.add_argument(
        "--masking_amount", default=0.1, type=float, help="Amount to heads to masking at each masking step."
    )
    parser.add_argument("--metric_name", default=None, type=str, help="Metric to use for head masking.")

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, sequences shorter padded.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup devices and distributed training
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend

    # Setup logging
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed: {}".format(args.device, args.n_gpu, bool(args.local_rank != -1)))

    # Set seeds
    set_seed(args)

    tracker = ImpactTracker(args.output_dir)
    tracker.launch_impact_monitor()

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    
    if args.metric_name is None:
        args.metric_name = {
            "cola": "mcc",
            "mnli": "acc",
            "mnli-mm": "acc",
            "mrpc": "acc",
            "sst-2": "acc",
            "sts-b": "corr",
            "qqp": "acc",
            "qnli": "acc",
            "rte": "acc",
            "wnli": "acc",
            "hans": "acc",
            "mnli_two": "acc",
            "hans_mnli": "acc"
        }[args.task_name]
    
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    MODEL_CLASSES["bert"] = (BertConfig, BertForSequenceClassification, MODEL_CLASSES["bert"][2])

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.model_type != "bert":
        raise NotImplemented("Not implemented for non BERT classes yet.")

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        output_attentions=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Print/save training arguments
    torch.save(args, os.path.join(args.output_dir, "run_args.bin"))
    logger.info("Training/evaluation parameters %s", args)

    # Prepare dataset for the GLUE task

    eval_data = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    true_eval_len = len(eval_data)
    
    if args.use_train_data:
        train_data = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        eval_data = random_split(train_data, [true_eval_len, len(train_data) - true_eval_len])[0]

    if args.data_subset > 0:
        eval_data = Subset(eval_data, list(range(min(args.data_subset, len(eval_data)))))

    eval_sampler = SequentialSampler(eval_data) if args.local_rank == -1 else DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    _, masked_amount = mask_global(args, model, eval_dataloader)

    # Generating a random mask of equal size
    fresh_model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    mask = prune_model(fresh_model, masked_amount, prune.RandomUnstructured)
    torch.save(mask, os.path.join(args.output_dir, "random_mask.p"))

    fresh_model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    mask = prune_model(fresh_model, masked_amount, L1UnstructuredInvert)
    torch.save(mask, os.path.join(args.output_dir, "bad_mask.pt"))

if __name__ == "__main__":
    main()
