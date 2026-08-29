from __future__ import annotations

import argparse
import json
from pathlib import Path

from mvbench_llava_anchor import write_json_atomic
from onevision_reader_quotient_equal_budget_protocol import (
    build_equal_budget_manifest,
    load_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvbench-manifest", type=Path, required=True)
    parser.add_argument("--source-fit-summary", type=Path, required=True)
    parser.add_argument("--videomme-manifest", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_equal_budget_manifest(
        mvbench_manifest=load_json(args.mvbench_manifest),
        source_fit_summary=load_json(args.source_fit_summary),
        videomme_manifest=load_json(args.videomme_manifest),
    )
    write_json_atomic(args.out_path, manifest)
    print(json.dumps(manifest["role_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
