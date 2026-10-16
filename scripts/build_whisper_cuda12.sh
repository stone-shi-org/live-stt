#!/usr/bin/env bash
# Build the whisper.cpp CUDA 12 backend bundle (see scripts/whisper-cuda12.Dockerfile).
# Needs only docker -- compiling needs no GPU/driver. Output: dist/*.tar.gz + sha256.
#   scripts/build_whisper_cuda12.sh [WHISPER_REF] [CUDA_ARCHS]
set -euo pipefail
cd "$(dirname "$0")/.."
REF="${1:-v1.9.4}"; ARCHS="${2:-61;75;80;86}"
mkdir -p dist
docker build -f scripts/whisper-cuda12.Dockerfile --target dist \
  --build-arg WHISPER_REF="$REF" --build-arg CUDA_ARCHS="$ARCHS" \
  --output type=local,dest=dist scripts
sha256sum dist/whisper-"$REF"-lib-linux-cuda12-x64.tar.gz
