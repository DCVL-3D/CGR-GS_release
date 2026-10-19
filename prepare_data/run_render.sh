#!/usr/bin/env bash
set -euo pipefail

RAW_ROOT=${RAW_ROOT:-data/thuman2.0_raw}
OUT_ROOT=${OUT_ROOT:-data/thuman2.0_rendered}
PACK_ROOT="${OUT_ROOT}/_rawpack_by_pid"
PHASES=${PHASES:-"train val"}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_PARALLEL=${NUM_PARALLEL:-2}
CAM_NUMS=16
SCENE_RADIUS=2.0
RES=2048

for phase in ${PHASES}; do
  total=$(ls -1 "${RAW_ROOT}/${phase}" | wc -l | tr -d ' ')
  batches=$(( (total + BATCH_SIZE - 1) / BATCH_SIZE ))
  for ((bi=0; bi<batches; bi+=NUM_PARALLEL)); do
    for ((i=0; i<NUM_PARALLEL; i++)); do
      cur=$((bi + i))
      if [ "${cur}" -lt "${batches}" ]; then
        python prepare_data/render_data.py --thuman_root "${RAW_ROOT}" --out_root "${OUT_ROOT}" \
          --phase "${phase}" --batch_size "${BATCH_SIZE}" --batch_idx "${cur}" \
          --cam_nums "${CAM_NUMS}" --scene_radius "${SCENE_RADIUS}" --res "${RES}" "${RES}" --skip_existing &
      fi
    done
    wait
    python prepare_data/unpack_raw.py --pack_root "${PACK_ROOT}" --out_root "${OUT_ROOT}" --phases "${phase}" --delete_packs
    find "${PACK_ROOT}/${phase}" -type d -empty -delete || true
  done
  python prepare_data/make_lr.py --root "${OUT_ROOT}" --phases "${phase}"
done
