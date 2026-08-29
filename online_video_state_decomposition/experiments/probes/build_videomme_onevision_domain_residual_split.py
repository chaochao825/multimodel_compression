from __future__ import annotations

import argparse
import json
from pathlib import Path

from videomme_onevision_domain_residual_protocol import (
    build_domain_residual_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import pyarrow.parquet as parquet

    records = parquet.read_table(args.parquet_path).to_pylist()
    source_split = json.loads(args.source_split.read_text(encoding="utf-8"))
    available_video_ids = {path.stem for path in args.video_root.glob("*.mp4")}
    manifest = build_domain_residual_manifest(
        source_split=source_split,
        records=records,
        available_video_ids=available_video_ids,
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_path.with_suffix(args.out_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out_path)
    print(json.dumps(manifest["role_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
