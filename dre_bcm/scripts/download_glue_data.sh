#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/glue_data}
TASKS=${TASKS:-SST-2 RTE}

mkdir -p "$DATA_DIR"

download_task() {
  local task="$1"
  local url="https://dl.fbaipublicfiles.com/glue/data/${task}.zip"
  local archive="${DATA_DIR}/${task}.zip"

  echo "downloading ${task} from ${url}"
  wget -q -O "$archive" "$url"
  unzip -oq "$archive" -d "$DATA_DIR"
}

for task in $TASKS; do
  download_task "$task"
done

echo "glue data ready under ${DATA_DIR}"
