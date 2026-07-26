#!/usr/bin/env bash
set -euo pipefail

device_number="${1:-20}"
card_label="${2:-linux-media-pipeline}"

sudo modprobe v4l2loopback \
  video_nr="${device_number}" \
  card_label="${card_label}" \
  exclusive_caps=1
