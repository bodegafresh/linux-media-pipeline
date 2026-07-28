#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup-loopback.sh [device_number] [card_label]
  ./scripts/setup-loopback.sh 20 lmp-horizontal 21 lmp-vertical

Examples:
  ./scripts/setup-loopback.sh
  ./scripts/setup-loopback.sh 20 linux-media-pipeline
  ./scripts/setup-loopback.sh 20 lmp-horizontal 21 lmp-vertical
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if (( $# == 0 )); then
  set -- 20 linux-media-pipeline
fi

if (( $# % 2 != 0 )); then
  echo "ERROR: expected device/card_label pairs." >&2
  usage >&2
  exit 1
fi

device_numbers=()
card_labels=()
while (( $# > 0 )); do
  device_number="$1"
  card_label="$2"
  shift 2

  if [[ ! "${device_number}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: device number must be numeric: ${device_number}" >&2
    exit 1
  fi

  device_numbers+=("${device_number}")
  card_labels+=("${card_label}")
done

join_by_comma() {
  local IFS=,
  echo "$*"
}

video_nr="$(join_by_comma "${device_numbers[@]}")"
card_label="$(join_by_comma "${card_labels[@]}")"
exclusive_caps="$(printf '1,%.0s' "${device_numbers[@]}")"
exclusive_caps="${exclusive_caps%,}"
recreate_command="./scripts/setup-loopback.sh"
for index in "${!device_numbers[@]}"; do
  recreate_command+=" ${device_numbers[index]} ${card_labels[index]}"
done

missing_before=()
for device_number in "${device_numbers[@]}"; do
  device_path="/dev/video${device_number}"
  if [[ ! -e "${device_path}" ]]; then
    missing_before+=("${device_path}")
  fi
done

if (( ${#missing_before[@]} > 0 )) && lsmod | grep -q '^v4l2loopback '; then
  echo "ERROR: v4l2loopback is already loaded and missing requested devices: ${missing_before[*]}" >&2
  echo "v4l2loopback devices are created from the module parameter list at load time." >&2
  echo "Close applications using loopback cameras, then recreate all needed devices together:" >&2
  echo "  sudo modprobe -r v4l2loopback" >&2
  echo "  ${recreate_command}" >&2
  exit 1
fi

sudo modprobe v4l2loopback \
  video_nr="${video_nr}" \
  card_label="${card_label}" \
  exclusive_caps="${exclusive_caps}"

missing_after=()
for device_number in "${device_numbers[@]}"; do
  device_path="/dev/video${device_number}"
  if [[ ! -e "${device_path}" ]]; then
    missing_after+=("${device_path}")
  fi
done

if (( ${#missing_after[@]} > 0 )); then
  echo "ERROR: requested loopback devices were not created: ${missing_after[*]}" >&2
  echo "Loaded v4l2loopback devices:" >&2
  v4l2-ctl --list-devices >&2 || true
  echo >&2
  echo "Try unloading stale loopback devices and creating them again:" >&2
  echo "  sudo modprobe -r v4l2loopback" >&2
  echo "  ${recreate_command}" >&2
  exit 1
fi

for index in "${!device_numbers[@]}"; do
  echo "Ready /dev/video${device_numbers[index]} (${card_labels[index]})"
done
v4l2-ctl --list-devices
