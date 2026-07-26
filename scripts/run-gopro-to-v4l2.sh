#!/usr/bin/env bash
set -euo pipefail

input_url="${1:-udp://0.0.0.0:8554?fifo_size=50000000&overrun_nonfatal=1}"
output_device="${2:-/dev/video20}"
width="${3:-1280}"
height="${4:-720}"
fps="${5:-30}"

if [[ ! -e "${output_device}" ]]; then
  device_number="${output_device#/dev/video}"
  ./scripts/setup-loopback.sh "${device_number}" linux-media-pipeline
fi

echo "Streaming GoPro UDP MPEG-TS to ${output_device}"
echo "Input: ${input_url}"
echo "Output: ${width}x${height}@${fps} RGB24"

exec ffmpeg \
  -hide_banner \
  -loglevel info \
  -fflags nobuffer \
  -flags low_delay \
  -strict experimental \
  -probesize 32 \
  -analyzeduration 0 \
  -i "${input_url}" \
  -an \
  -vf "fps=${fps},scale=${width}:${height}:flags=fast_bilinear,format=rgb24" \
  -pix_fmt rgb24 \
  -f v4l2 \
  "${output_device}"
