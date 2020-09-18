# Evaluate all models and compute metrics with mean and std across seeds.

python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/no_mask

# Importance Pruning Results

## (good pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --head_masks_dir ../masks/heads_mlps --mlp_masks_dir ../masks/heads_mlps  --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp

## (bad pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --head_masks_dir ../masks/heads_mlps --mlp_masks_dir ../masks/heads_mlps --mask_mode bad  --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_bad

## (random pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --head_masks_dir ../masks/heads_mlps --mlp_masks_dir ../masks/heads_mlps --mask_mode random  --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_random


## (good retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_heads_mlps/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_retrained

## (bad retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_heads_mlps_bad/ --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_bad_retrained

## (random retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_heads_mlps/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_random_retrained


# Magnitude Pruning Results


## (good pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --global_masks_dir ../masks/global --global_mask_file_name magnitude_mask.p  --do_lower_case --model_type bert --output_dir ../evaluate_masked/global_magnitude

## (bad pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --global_masks_dir ../masks/global --global_mask_file_name bad_mask.pt  --do_lower_case --model_type bert --output_dir ../evaluate_masked/global_bad

## (random pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --global_masks_dir ../masks/global --global_mask_file_name random_mask.p --do_lower_case --model_type bert --output_dir ../evaluate_masked/global_random


## (good retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_global_magnitude_mask/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/global_magnitude_retrained

## (bad retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_global_bad_mask/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/global_bad_retrained

## (random retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_global_random_mask/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/global_random_retrained


# Importance Pruning - Super Results

## (good pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --head_masks_dir ../masks/heads_mlps_super --mlp_masks_dir ../masks/heads_mlps_super  --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_super

## (bad pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --head_masks_dir ../masks/heads_mlps_super_bizzaro --mlp_masks_dir ../masks/heads_mlps_super_bizzaro  --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_super_bizzaro

## (random pruned)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned/ --head_masks_dir ../masks/heads_mlps_super_midling --mlp_masks_dir ../masks/heads_mlps_super_midling   --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_super_midling


## (good retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_heads_mlps_super/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_super_retrained

## (bad retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_heads_mlps_super_bizzaro/ --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_super_bizzaro_retrained

## (random retrained)
python franken_bert.py --data_dir ../data/glue/ --models_dir ../models/finetuned_heads_mlps_super_midling/   --do_lower_case --model_type bert --output_dir ../evaluate_masked/head_mlp_super_midling_retrained
