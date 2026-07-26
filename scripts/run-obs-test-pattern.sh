#!/usr/bin/env bash
set -euo pipefail

device_number="${1:-20}"
card_label="${2:-linux-media-pipeline}"
device_path="/dev/video${device_number}"

if [[ ! -e "${device_path}" ]]; then
  ./scripts/setup-loopback.sh "${device_number}" "${card_label}"
fi

./scripts/build.sh
echo "Streaming OBS test pattern to ${device_path}. Press Ctrl+C to stop."
./build/dev/lmp --test-pattern
