#!/usr/bin/env bash
set -euo pipefail

input_device="${1:-/dev/video0}"
output_device="${2:-/dev/video20}"
width="${3:-1280}"
height="${4:-720}"
fps="${5:-30}"

if [[ ! -e "${output_device}" ]]; then
  device_number="${output_device#/dev/video}"
  ./scripts/setup-loopback.sh "${device_number}" linux-media-pipeline
fi

echo "Streaming USB camera ${input_device} to ${output_device}"
echo "Output: ${width}x${height}@${fps} RGB24"

exec ffmpeg \
  -hide_banner \
  -loglevel info \
  -fflags nobuffer \
  -f v4l2 \
  -framerate "${fps}" \
  -video_size "${width}x${height}" \
  -i "${input_device}" \
  -an \
  -vf "fps=${fps},scale=${width}:${height}:flags=fast_bilinear,format=rgb24" \
  -pix_fmt rgb24 \
  -f v4l2 \
  "${output_device}"
