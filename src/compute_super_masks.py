import numpy as np
import pathlib
import matplotlib.pyplot as plt
import time

import pathlib
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import random
def load_head_data(experiments_path):
    head_data = {}
    for task_dir in experiments_path.iterdir():
        head_data[task_dir.stem] = {}
        for seed_dir in task_dir.iterdir():
            head_mask = np.load(seed_dir / "head_mask.npy")
            head_data[task_dir.stem][seed_dir.stem] = {
                "head_mask": head_mask,
            }
    return head_data
def load_mlp_data(experiments_path):
    mlp_data = {}
    for task_dir in experiments_path.iterdir():
        mlp_data[task_dir.stem] = {}
        for seed_dir in task_dir.iterdir():
            mlp_mask = np.load(seed_dir / "mlp_mask.npy")
            mlp_importance = np.load(seed_dir / "mlp_importance.npy")
            mlp_data[task_dir.stem][seed_dir.stem] = {
                "mlp_mask": mlp_mask,
                "mlp_importance": mlp_importance
            }
    return mlp_data
def sample_bad_to_similar_size(bad, good):
    total_good = good.sum()
    total_bad = bad.sum()
    bad = bad.copy()
    if total_good > total_bad:
        to_add = total_good - total_bad
        
        selection = np.argwhere((good == 0) & (bad == 0)).tolist()
        if len(selection) < to_add:
            selection = np.argwhere(bad == 0).tolist()
        
        add_indices = random.sample(selection, int(to_add))
        for idx in add_indices:
            if len(idx) == 2:
                bad[idx[0], idx[1]] = 1
            else:
                bad[idx[0]] = 1
    elif total_good < total_bad:
        to_remove = total_bad - total_good
        remove_indices = random.sample(np.argwhere(bad == 1).tolist(), int(to_remove))
        for idx in remove_indices:
            if len(idx) == 2:
                bad[idx[0], idx[1]] = 0
            else:
                bad[idx[0]] = 0
    return bad

def sample_random_middlings(bad, good):
    midling = np.ones_like(good)
    midling[good == 1] = 0
    midling[bad == 1] = 0
    
    if midling.sum() >= good.sum():
        to_remove = midling.sum() - good.sum()
        remove_indices = random.sample(np.argwhere(midling == 1).tolist(), int(to_remove))
        for idx in remove_indices:
            if len(idx) == 2:
                midling[idx[0], idx[1]] = 0
            else:
                midling[idx[0]] = 0
    else:
        to_add = good.sum() - midling.sum()
        add_indices = random.sample(np.argwhere(midling == 0).tolist(), int(to_add))
        for idx in add_indices:
            if len(idx) == 2:
                midling[idx[0], idx[1]] = 1
            else:
                midling[idx[0]] = 1
        
    return midling

if __name__ == '__main__':
    random.seed(1337)
    experiments_path = pathlib.Path("../masks/heads_mlps")
    together_head_data = load_head_data(experiments_path)
    experiments_path = pathlib.Path("../masks/heads_mlps")
    together_mlp_data = load_mlp_data(experiments_path)

    super_survivior_ids = {}

    super_save_path = pathlib.Path("../masks/heads_mlps_super")
    super_bad_save_path = pathlib.Path("../masks/heads_mlps_super_bizzaro")
    super_midling_save_path = pathlib.Path("../masks/heads_mlps_super_midling")

    for task in sorted(together_head_data.keys()):
        print(f"{task}")

        super_head = np.ones((12, 12))
        super_mlp = np.ones((12,))
        super_bad_head = np.ones((12, 12))
        super_bad_mlp = np.ones((12,))
        for seed in together_head_data[task]:
            super_head *= together_head_data[task][seed]["head_mask"]
            super_mlp *= together_mlp_data[task][seed]["mlp_mask"]
            super_bad_head *= (1 - together_head_data[task][seed]["head_mask"])
            super_bad_mlp *= (1 - together_mlp_data[task][seed]["mlp_mask"])
        super_survivior_ids[task] = {
            "head": [x[0] for x in np.argwhere(super_head.reshape(-1) == 1)],
            "mlp": [x[0] for x in np.argwhere(super_mlp.reshape(-1) == 1)]
        }
        
        
        print(f"Super heads: {super_head.sum()}")
        print(f"Super MLPs:  {super_mlp.sum()}")
        
        
        for seed in together_head_data[task]:
            
            super_path = super_save_path / task / seed
            bizzaro_path = super_bad_save_path / task / seed
            midling_path = super_midling_save_path / task / seed
            
            super_path.mkdir(parents=True, exist_ok=True)
            bizzaro_path.mkdir(parents=True, exist_ok=True)
            midling_path.mkdir(parents=True, exist_ok=True)
            np.save(str(super_path / "head_mask.npy"), super_head)
            np.save(str(super_path / "mlp_mask.npy"), super_mlp)   

            super_bad_head = sample_bad_to_similar_size(super_bad_head, super_head)
            super_bad_mlp = sample_bad_to_similar_size(super_bad_mlp, super_mlp)
            
            print(super_bad_head.sum(), "Bizzaro Overlaps with super head:", (super_bad_head * super_head).sum())
            print(super_bad_mlp.sum(),   "Bizzaro Overlaps with super mlp:", (super_bad_mlp * super_mlp).sum())
            
            np.save(str(bizzaro_path / "head_mask.npy"), super_bad_head)
            np.save(str(bizzaro_path / "mlp_mask.npy"), super_bad_mlp)        
            
            super_midling_head = sample_random_middlings(super_bad_head, super_head)
            super_midling_mlp = sample_random_middlings(super_bad_mlp, super_mlp)        
            print(super_midling_head.sum(),  "Midling Overlaps with super head:", (super_midling_head * super_head).sum())
            print(super_midling_mlp.sum(),"Midling Overlaps with super mlp:", (super_midling_mlp * super_mlp).sum())
        
            np.save(str(midling_path / "head_mask.npy"), super_midling_head)
            np.save(str(midling_path / "mlp_mask.npy"), super_midling_mlp)