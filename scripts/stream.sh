#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/stream.sh test-pattern [output_device] [config] [lmp_args...]
  ./scripts/stream.sh gopro-udp [output_device] [config] [lmp_args...]
  ./scripts/stream.sh install-model
  ./scripts/stream.sh usb [input_device] [output_device] [width] [height] [fps]

Examples:
  ./scripts/stream.sh test-pattern
  ./scripts/stream.sh gopro-udp
  ./scripts/stream.sh gopro-udp /dev/video20 config/presets/clean.yaml
  ./scripts/stream.sh gopro-udp /dev/video20 config/presets/ai-background-blur.yaml --stats-every 5
  ./scripts/stream.sh install-model
  ./scripts/stream.sh usb /dev/video0 /dev/video20 1280 720 30
EOF
}

ensure_loopback() {
  local output_device="$1"
  if [[ -e "${output_device}" ]]; then
    return
  fi

  local device_number="${output_device#/dev/video}"
  if [[ "${device_number}" == "${output_device}" ]]; then
    echo "ERROR: output device must look like /dev/videoN: ${output_device}" >&2
    exit 1
  fi

  ./scripts/setup-loopback.sh "${device_number}" linux-media-pipeline
}

mode="${1:-}"
if [[ -z "${mode}" || "${mode}" == "--help" || "${mode}" == "-h" ]]; then
  usage
  exit 0
fi
shift

case "${mode}" in
  install-model)
    model_dir="assets/models"
    model_path="${model_dir}/person-segmentation.onnx"
    model_zip_url="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/mediapipe_selfie/releases/v0.52.0/mediapipe_selfie-onnx-float.zip"
    fallback_model_url="https://unpkg.com/jp.ikep.mediapipe.selfiesegmentation@1.0.1/ONNX/selfie_segmentation.onnx"
    tmp_zip="$(mktemp)"
    tmp_dir="$(mktemp -d)"
    mkdir -p "${model_dir}"
    echo "Downloading MediaPipe Selfie Segmentation ONNX model..."
    if curl -fL "${model_zip_url}" -o "${tmp_zip}"; then
      unzip -q -o "${tmp_zip}" -d "${tmp_dir}"
      cp "${tmp_dir}/mediapipe_selfie-onnx-float/mediapipe_selfie.onnx" "${model_path}"
      cp "${tmp_dir}/mediapipe_selfie-onnx-float/mediapipe_selfie.data" "${model_dir}/mediapipe_selfie.data"
    else
      echo "Preferred model download failed; using fallback ONNX model." >&2
      curl -fL "${fallback_model_url}" -o "${model_path}"
    fi
    rm -rf "${tmp_zip}" "${tmp_dir}"
    echo "Installed ${model_path}"
    ;;

  test-pattern)
    output_device="${1:-/dev/video20}"
    config="${2:-config/default.yaml}"
    shift $(( $# >= 1 ? 1 : 0 ))
    shift $(( $# >= 1 ? 1 : 0 ))
    ensure_loopback "${output_device}"
    ./scripts/build.sh
    echo "Streaming test pattern to ${output_device}. Press Ctrl+C to stop."
    echo "Runtime settings come from ${config}."
    exec ./build/dev/lmp --config "${config}" --test-pattern "$@"
    ;;

  gopro-udp)
    output_device="${1:-/dev/video20}"
    config="${2:-config/default.yaml}"
    shift $(( $# >= 1 ? 1 : 0 ))
    shift $(( $# >= 1 ? 1 : 0 ))
    ensure_loopback "${output_device}"
    ./scripts/build.sh
    echo "Streaming configured GoPro UDP capture through C++ pipeline to ${output_device}."
    echo "Runtime settings come from ${config}."
    exec ./build/dev/lmp --config "${config}" --stream-live "$@"
    ;;

  usb)
    input_device="${1:-/dev/video0}"
    output_device="${2:-/dev/video20}"
    width="${3:-1280}"
    height="${4:-720}"
    fps="${5:-30}"
    ensure_loopback "${output_device}"
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
    ;;

  *)
    echo "ERROR: unknown stream mode: ${mode}" >&2
    usage >&2
    exit 1
    ;;
esac
