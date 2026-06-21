#!/usr/bin/env bash
#
# Download the KITTI raw drives used by the default split (configs/experiment/default.yaml).
#
#   bash deploy/vast/download_kitti.sh [DEST_DIR]
#   PARALLEL=8 bash deploy/vast/download_kitti.sh /workspace/kitti_raw
#
# Pulls the sync archives for the train/val drives (2011_09_26) and the test drive
# (2011_09_30/0028), plus the per-date calibration archives, and unzips them into the
# KITTI raw layout the dataset loader expects:
#
#   $DEST_DIR/2011_09_26/2011_09_26_drive_0001_sync/{image_02,velodyne_points}/...
#   $DEST_DIR/2011_09_26/calib_cam_to_cam.txt , calib_velo_to_cam.txt , ...
#
# Notes on speed/robustness (the KITTI mirror is S3 in Frankfurt — a single stream
# from a distant host is latency-bound, not bandwidth-bound):
#   * downloads run in parallel (PARALLEL, default 6) to hide round-trip latency;
#   * each transfer is bounded (10-min cap, 30s connect timeout, 5 retries, resume)
#     so a stalled connection can never hang the whole job;
#   * each archive is deleted right after extraction to keep peak disk usage low.
# A host geographically close to Frankfurt downloads noticeably faster.
set -uo pipefail   # not -e: a single drive failure must not abort the whole batch

DEST_DIR="${1:-${DATA_DIR:-/workspace/kitti_raw}}"
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
PARALLEL="${PARALLEL:-6}"

mkdir -p "$DEST_DIR"

# Train + val drives (date 2011_09_26) and the test drive (date 2011_09_30 / 0028).
DRIVES_0926=(0001 0002 0005 0009 0011 0013 0014 0015 0017 0018 0019 0020 0022 0023 \
             0027 0028 0029 0032 0035 0036 0039 0046 0048 0051 0052 0056 0057 0059 \
             0060 0061 0064 0070 0079 0084 0086 0087 0091 0093 0095 0096 0101 0104 \
             0106 0113 0117)
DRIVES_0930=(0028)

# Download + extract a single archive (path relative to BASE_URL). Safe to run in
# parallel: each call uses its own temp zip and extracts into the shared DEST_DIR.
fetch_one () {
  local path="$1"
  local url="$BASE_URL/$path"
  local zip="$DEST_DIR/.dl_${path//\//_}"
  if timeout 600 wget -q --timeout=30 --tries=5 -c "$url" -O "$zip"; then
    if unzip -n -q -d "$DEST_DIR" "$zip"; then
      echo "[kitti] OK    ${path##*/}"
    else
      echo "[kitti] UNZIP ${path##*/} (corrupt/partial)"
    fi
  else
    echo "[kitti] FAIL  ${path##*/} (download timed out / errored)"
  fi
  rm -f "$zip"
}
export -f fetch_one
export BASE_URL DEST_DIR

# Assemble the work list, then fan it out across PARALLEL workers.
LIST=(2011_09_26_calib.zip 2011_09_30_calib.zip)
for d in "${DRIVES_0926[@]}"; do LIST+=("2011_09_26_drive_${d}/2011_09_26_drive_${d}_sync.zip"); done
for d in "${DRIVES_0930[@]}"; do LIST+=("2011_09_30_drive_${d}/2011_09_30_drive_${d}_sync.zip"); done

echo "[kitti] downloading ${#LIST[@]} archives with PARALLEL=$PARALLEL → $DEST_DIR"
printf '%s\n' "${LIST[@]}" | xargs -P "$PARALLEL" -I {} bash -c 'fetch_one "$@"' _ {}

n26=$(ls "$DEST_DIR/2011_09_26" 2>/dev/null | grep -c drive || true)
n30=$(ls "$DEST_DIR/2011_09_30" 2>/dev/null | grep -c drive || true)
echo "[kitti] done → $DEST_DIR (2011_09_26=$n26 drives, 2011_09_30=$n30 drives)"
