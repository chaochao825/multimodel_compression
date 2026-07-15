import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_best_metric(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["source_file"] = str(path)
    payload["result_dir"] = str(path.parent)
    return payload


def collect(root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for path in sorted(root.rglob("best_metrics.json")):
        try:
            rows.append(load_best_metric(path))
        except Exception as exc:  # pragma: no cover
            rows.append(
                {
                    "source_file": str(path),
                    "result_dir": str(path.parent),
                    "error": str(exc),
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/matrix_fit")
    parser.add_argument("--output", default="results/matrix_fit/best_metrics_summary.csv")
    args = parser.parse_args()

    rows = collect(Path(args.root))
    write_csv(Path(args.output), rows)
    print(f"collected {len(rows)} best-metric files into {args.output}")


if __name__ == "__main__":
    main()
