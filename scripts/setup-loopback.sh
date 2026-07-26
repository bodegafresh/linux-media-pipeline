#!/usr/bin/env bash
set -euo pipefail

device_number="${1:-20}"
card_label="${2:-linux-media-pipeline}"

sudo modprobe v4l2loopback \
  video_nr="${device_number}" \
  card_label="${card_label}" \
  exclusive_caps=1

device_path="/dev/video${device_number}"

if [[ ! -e "${device_path}" ]]; then
  echo "ERROR: ${device_path} was not created." >&2
  echo "Loaded v4l2loopback devices:" >&2
  v4l2-ctl --list-devices >&2 || true
  echo >&2
  echo "Try unloading stale loopback devices and creating it again:" >&2
  echo "  sudo modprobe -r v4l2loopback" >&2
  echo "  ./scripts/setup-loopback.sh ${device_number} ${card_label}" >&2
  exit 1
fi

echo "Created ${device_path} (${card_label})"
v4l2-ctl --list-devices
