import os
import argparse
import re
import yaml
import numpy as np

from pathlib import Path
from track_by_gt_bbox import main as track_by_gt_bbox


def extend_dict(extend_me, extend_by):
    if isinstance(extend_by, dict):
        for k, v in extend_by.items():
            if k in extend_me:
                extend_dict(extend_me.get(k), v)
            else:
                extend_me[k] = v
    else:
        extend_me += extend_by


def main(config, config_iou, config_sort, config_deepsort):    
    trackers = ["IOU", "SORT", "DEEP_SORT"]
    max_iou_dists = [float(i) / 10 for i in range(1, 10)]

    regex = re.compile(r'\d+')
    exp_ids = [[int(x) for x in regex.findall(x)] for x in os.listdir("configs")]
    if any(exp_ids):
        exp_id = np.max(exp_ids) + 1
    else:
        exp_id = 0

    for tracker in trackers:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)

        if tracker == "IOU":
            with open(config_iou, 'r') as f:
                tracker_cfg = yaml.safe_load(f)
        elif tracker == "SORT":
            with open(config_sort, 'r') as f:
                tracker_cfg = yaml.safe_load(f)
        elif tracker == "DEEP_SORT":
            with open(config_deepsort, 'r') as f:
                tracker_cfg = yaml.safe_load(f)

        extend_dict(cfg, tracker_cfg)

        print(cfg)

        for max_iou_dist in max_iou_dists:
            cfg["EXP_ID"] = str(exp_id)
            cfg["TRACKER"] = tracker.lower()
            cfg[tracker]["MAX_IOU_DISTANCE"] = max_iou_dist

            output_file = os.path.join("configs", f"exp_{exp_id}.yaml")
            with open(output_file, 'w', encoding='utf8') as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

            track_by_gt_bbox(output_file)

            exp_id += 1
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-iou", type=Path, required=True)
    parser.add_argument("--config-sort", type=Path, required=True)
    parser.add_argument("--config-deepsort", type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.config_iou, args.config_sort, args.config_deepsort)