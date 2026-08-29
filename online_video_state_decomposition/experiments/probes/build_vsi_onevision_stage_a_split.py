from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvbench_llava_anchor import write_json_atomic
from vsi_onevision_protocol import build_vsi_scene_split, load_vsi_mcq_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    split = build_vsi_scene_split(records)
    write_json_atomic(args.out_path, split)
    print(json.dumps(split["role_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
