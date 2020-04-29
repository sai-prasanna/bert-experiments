# !mkdir -p ../bad_head_mlp_masks_results
# import os
# import json
# evaluation_dir = pathlib.Path("../finetuned_models_with_head_and_mlp_mask_bad")
# output_dir = pathlib.Path("../bad_head_mlp_masks_results")
# for task_dir in evaluation_dir.iterdir():
#     for seed in task_dir.iterdir():
#         with (seed / "config.json").open() as f:
#             config = json.load(f)
#             pruned_heads = config["pruned_heads"]
#             pruned_mlps = config["pruned_heads"]
#             head_mask = np.ones((12, 12))
#             mlp_mask = np.ones(12)
#             for layer, heads in pruned_heads.items():
#                 for head in heads:
#                     head_mask[int(layer)][head] = 0
#             for layer in pruned_mlps:
#                 mlp_mask[int(layer)] = 0
#             mask_dir = output_dir / task_dir.stem / seed.stem
#             os.makedirs(str(mask_dir))
#             np.save(str(mask_dir / "head_mask.npy"), head_mask)
#             np.save(str(mask_dir / "mlp_mask.npy"), mlp_mask)
