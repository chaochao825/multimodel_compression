#!/usr/bin/env bash
set -u

roots=(
  /data6/zmf/miniconda3
  /data6/anaconda3
  /data2/wangmeiqi/anaconda3
  /data6/phz/envs
)

seen=""
for root in "${roots[@]}"; do
  if [[ -x "${root}/bin/python" ]]; then
    candidates=("${root}/bin/python")
  else
    candidates=()
  fi
  if [[ -d "${root}/envs" ]]; then
    while IFS= read -r py; do
      candidates+=("${py}")
    done < <(find "${root}/envs" -maxdepth 3 \( -type f -o -type l \) -path "*/bin/python" 2>/dev/null | sort)
  fi
  if [[ "${root}" == */envs ]]; then
    while IFS= read -r py; do
      candidates+=("${py}")
    done < <(find "${root}" -maxdepth 3 \( -type f -o -type l \) -path "*/bin/python" 2>/dev/null | sort)
  fi

  for py in "${candidates[@]}"; do
    case "${seen}" in
      *"|${py}|"*) continue ;;
    esac
    seen="${seen}|${py}|"
    "${py}" - <<'PY' 2>/dev/null
import json
import sys

row = {"python": sys.executable, "version": sys.version.split()[0]}
for name in ("torch", "diffusers", "transformers", "safetensors"):
    try:
        mod = __import__(name)
        row[name] = getattr(mod, "__version__", "ok")
    except Exception as exc:
        row[name] = "ERR:" + exc.__class__.__name__
try:
    import torch
    row["cuda_available"] = bool(torch.cuda.is_available())
    row["cuda_count"] = int(torch.cuda.device_count())
except Exception:
    row["cuda_available"] = False
    row["cuda_count"] = 0
print(json.dumps(row, ensure_ascii=False, sort_keys=True))
PY
  done
done
