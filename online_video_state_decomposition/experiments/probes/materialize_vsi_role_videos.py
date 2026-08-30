from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from mvbench_llava_anchor import write_json_atomic
from vsi_onevision_protocol import PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("calibration", "selection"), required=True)
    return parser.parse_args()


def materialize_role(
    *,
    split: dict[str, object],
    archive_root: Path,
    out_dir: Path,
    role: str,
) -> dict[str, object]:
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    if role not in split["roles"]:
        raise ValueError(f"unknown VSI split role: {role}")

    members_by_dataset: dict[str, list[str]] = defaultdict(list)
    for scene in split["roles"][role]:
        dataset = str(scene["dataset"])
        relative_path = Path(str(scene["relative_video_path"]))
        if relative_path.parts[0] != dataset or relative_path.suffix != ".mp4":
            raise ValueError(f"invalid VSI video path: {relative_path}")
        members_by_dataset[dataset].append(relative_path.as_posix())

    materialized = []
    reused = []
    for dataset, members in sorted(members_by_dataset.items()):
        archive_path = archive_root / f"{dataset}.zip"
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            available = set(archive.namelist())
            for member in sorted(members):
                if member not in available:
                    raise FileNotFoundError(f"{member} is absent from {archive_path}")
                destination = out_dir / Path(member)
                if destination.is_file():
                    reused.append(member)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".mp4.partial")
                if temporary.exists():
                    raise FileExistsError(temporary)
                with archive.open(member) as source, temporary.open("xb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                temporary.replace(destination)
                materialized.append(member)

    expected_count = len(split["roles"][role])
    if len(materialized) + len(reused) != expected_count:
        raise RuntimeError("VSI materialized video count mismatch")
    return {
        "protocol_id": PROTOCOL_ID,
        "role": role,
        "expected_count": expected_count,
        "materialized_count": len(materialized),
        "reused_count": len(reused),
        "materialized_paths": materialized,
        "reused_paths": reused,
    }


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    summary = materialize_role(
        split=split,
        archive_root=args.archive_root,
        out_dir=args.out_dir,
        role=args.role,
    )
    write_json_atomic(args.out_dir / f"materialize_{args.role}.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
