#!/usr/bin/env bash
#
# Fetch a parakeet.cpp GGUF into ./models/ with a sha256 check -- a truncated
# download otherwise surfaces later as a confusing ggml abort() at model load
# time, not as a clear "the file is wrong" error here.
#
#   ./scripts/fetch_model.sh                          both default models
#   ./scripts/fetch_model.sh realtime_eou_120m-v1      just the default
#   ./scripts/fetch_model.sh nemotron-3.5-asr-streaming-0.6b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

MODELS_DIR="models"
HF_REPO="mudler/parakeet-cpp-gguf"
mkdir -p "$MODELS_DIR"

# key -> "filename sha256". sha256 pulled from HF's x-linked-etag header for
# the q8_0 file (verified against a fully-downloaded copy for the default
# model; see CLAUDE.md).
declare -A MODEL_FILES=(
    ["realtime_eou_120m-v1"]="realtime_eou_120m-v1-q8_0.gguf 62616b914d6f5a683a5dea672df055b57de5c49dddf871b8b44b9c814dc3d896"
    ["nemotron-3.5-asr-streaming-0.6b"]="nemotron-3.5-asr-streaming-0.6b-q8_0.gguf ba2f13eccd4a5245be728f77e6149bd6a4fdcdd133ff2e08ac6005bcef7a99f1"
)

fetch_one() {
    local key="$1"
    local entry="${MODEL_FILES[$key]:-}"
    if [ -z "$entry" ]; then
        echo "ERROR: unknown model key '$key'. Known: ${!MODEL_FILES[*]}" >&2
        return 1
    fi
    local filename="${entry% *}"
    local expected_sha="${entry#* }"
    local dest="$MODELS_DIR/$filename"
    local url="https://huggingface.co/${HF_REPO}/resolve/main/${filename}"

    if [ -f "$dest" ]; then
        local actual_sha
        actual_sha="$(sha256sum "$dest" | cut -d' ' -f1)"
        if [ "$actual_sha" = "$expected_sha" ]; then
            echo "==> $filename already present and verified, skipping"
            return 0
        fi
        echo "==> $filename present but checksum mismatch, re-fetching"
    fi

    echo "==> Fetching $filename from $url"
    curl -fL --progress-bar -o "$dest.tmp" "$url"

    local actual_sha
    actual_sha="$(sha256sum "$dest.tmp" | cut -d' ' -f1)"
    if [ "$actual_sha" != "$expected_sha" ]; then
        echo "ERROR: sha256 mismatch for $filename" >&2
        echo "  expected: $expected_sha" >&2
        echo "  actual:   $actual_sha" >&2
        rm -f "$dest.tmp"
        return 1
    fi
    mv "$dest.tmp" "$dest"
    echo "==> $filename verified ($actual_sha)"
}

if [ $# -eq 0 ]; then
    for key in "${!MODEL_FILES[@]}"; do
        fetch_one "$key"
    done
else
    for key in "$@"; do
        fetch_one "$key"
    done
fi
