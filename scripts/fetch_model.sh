#!/usr/bin/env bash
#
# Fetch a model file into ./models/ with a sha256 check -- a truncated
# download otherwise surfaces later as a confusing ggml/whisper abort() at
# model load time, not as a clear "the file is wrong" error here.
#
#   ./scripts/fetch_model.sh                          the two parakeet-family defaults
#   ./scripts/fetch_model.sh realtime_eou_120m-v1      just the default
#   ./scripts/fetch_model.sh nemotron-3.5-asr-streaming-0.6b
#   ./scripts/fetch_model.sh whisper-base.en-q8_0      whisper family, fetched explicitly
#   ./scripts/fetch_model.sh whisper-large-v3-turbo-q8_0
#
# No-args behavior deliberately stays scoped to the two original (small,
# parakeet-family) defaults -- see DEFAULT_KEYS below -- rather than also
# pulling in the whisper family's much larger files (up to ~1GB) every time
# this is run with no arguments. Fetch a whisper model explicitly by key.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

# key -> "repo filename sha256". Two separate HF repos/families:
#   - mudler/parakeet-cpp-gguf: parakeet.cpp's own GGUF conversions.
#   - ggerganov/whisper.cpp: whisper's own legacy ggml *.bin format (NOT
#     GGUF -- confirmed against the real repo listing; see CLAUDE.md and
#     live_stt/models.py's comment on ModelSpec.gguf_filename).
#
# sha256 pulled from HF's x-linked-etag header for each file (verified
# against a fully-downloaded copy for realtime_eou_120m-v1 and
# whisper-base.en-q8_0 -- the two actually run through this script while
# building whisper support; the rest are header-sourced only, same
# methodology this file already used for nemotron before this change).
#
# q8_0 does not exist for every whisper size upstream -- large-v3 only has
# q5_0 (confirmed live: ggml-large-v3-q8_0.bin 404s), so that entry uses the
# next-closest quantization instead rather than the unquantized ~3GB file.
declare -A MODEL_FILES=(
    ["realtime_eou_120m-v1"]="mudler/parakeet-cpp-gguf realtime_eou_120m-v1-q8_0.gguf 62616b914d6f5a683a5dea672df055b57de5c49dddf871b8b44b9c814dc3d896"
    ["nemotron-3.5-asr-streaming-0.6b"]="mudler/parakeet-cpp-gguf nemotron-3.5-asr-streaming-0.6b-q8_0.gguf ba2f13eccd4a5245be728f77e6149bd6a4fdcdd133ff2e08ac6005bcef7a99f1"
    ["whisper-base.en-q8_0"]="ggerganov/whisper.cpp ggml-base.en-q8_0.bin a4d4a0768075e13cfd7e19df3ae2dbc4a68d37d36a7dad45e8410c9a34f8c87e"
    ["whisper-small-q8_0"]="ggerganov/whisper.cpp ggml-small-q8_0.bin 49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f"
    ["whisper-medium-q8_0"]="ggerganov/whisper.cpp ggml-medium-q8_0.bin 42a1ffcbe4167d224232443396968db4d02d4e8e87e213d3ee2e03095dea6502"
    ["whisper-large-v3-q5_0"]="ggerganov/whisper.cpp ggml-large-v3-q5_0.bin d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1"
    ["whisper-large-v3-turbo-q8_0"]="ggerganov/whisper.cpp ggml-large-v3-turbo-q8_0.bin 317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1"
)

# No-args default set -- see the file header comment on why this doesn't
# just mean "every key in MODEL_FILES".
DEFAULT_KEYS=("realtime_eou_120m-v1" "nemotron-3.5-asr-streaming-0.6b")

fetch_one() {
    local key="$1"
    local entry="${MODEL_FILES[$key]:-}"
    if [ -z "$entry" ]; then
        echo "ERROR: unknown model key '$key'. Known: ${!MODEL_FILES[*]}" >&2
        return 1
    fi
    local repo="${entry%% *}"
    local rest="${entry#* }"
    local filename="${rest% *}"
    local expected_sha="${rest#* }"
    local dest="$MODELS_DIR/$filename"
    local url="https://huggingface.co/${repo}/resolve/main/${filename}"

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
    for key in "${DEFAULT_KEYS[@]}"; do
        fetch_one "$key"
    done
else
    for key in "$@"; do
        fetch_one "$key"
    done
fi
