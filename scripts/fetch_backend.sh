#!/usr/bin/env bash
#
# LS-5 Phase 7b: fetch a prebuilt ENGINE artifact (the C++ library/binaries
# themselves) from the real upstream GitHub release, instead of building
# parakeet.cpp/whisper.cpp from source. Sibling to scripts/fetch_model.sh,
# same download -> sha256-verify -> extract -> skip-if-present shape --
# deliberately PURE BASH, not a call into the live_stt Python package, for
# the exact same reason fetch_model.sh is: this needs to run in an early
# Docker build layer (worker/main*.cpp's compile step) before the Python
# venv/`pip install -r requirements.txt` layer even exists. This DOES mean
# the (repo, tag, filename, sha256) table below is a second copy of
# live_stt/backends/registry.py's data, same as fetch_model.sh already is
# a second copy of live_stt/models.py's -- keep the two in sync by hand,
# there is no single source of truth today (same tradeoff, not a new one).
#
#   ./scripts/fetch_backend.sh parakeet cpu [dest_dir]   default dest: ./backends/parakeet/cpu
#   ./scripts/fetch_backend.sh whisper cpu [dest_dir]    default dest: ./backends/whisper/cpu
#
# Real GitHub release artifacts, verified for real while building this (see
# the LS-5 Jira ticket for the nm -D/ldd investigation) -- NOT invented
# repo names or guessed checksums. sha256 computed from an actual full
# download of each tarball, the same way fetch_model.sh's own comment
# describes for its two originally-verified entries.
#
# CUDA flavors are deliberately NOT registered here yet -- a real cuda
# entry needs a full download (500MB-900MB+ per file) and a real GPU to
# confirm it actually runs, neither of which was done in this pass. Add
# them following this exact pattern once that's actually been done, with a
# freshly computed sha256, not a guess.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# key = "<engine>:<flavor>" -> "repo tag asset_filename sha256"
declare -A BACKEND_FILES=(
    ["parakeet:cpu"]="mudler/parakeet.cpp v0.5.0 parakeet-v0.5.0-lib-linux-cpu-x64.tar.gz 07cc1095924a134fe5dc7e6e44247fea4eada70ba56a663f73281b96da0624f2"
    # ggerganov/whisper.cpp's "bin" release (no separate lib+headers package
    # for Linux -- see live_stt/backends/registry.py's docstring) ships
    # libwhisper.so + a set of per-microarch libggml-cpu-<name>.so variants,
    # plus example binaries this project doesn't use. Fetched and verified
    # for real (this exact tarball, this exact sha256), but NOT YET wired
    # into worker/CMakeLists.txt's build (that engine's from-source static
    # build is unchanged this pass) -- registered here so the mapping is in
    # one place ready for that follow-up, not because it's consumed today.
    ["whisper:cpu"]="ggerganov/whisper.cpp b5130 whisper-bin-ubuntu-x64.tar.gz 53e7fd8b5764edad916b8848dd0af6abb1ff1d3b86c899e79c78652412536c32"
)

if [ $# -lt 2 ]; then
    echo "usage: $0 <engine> <flavor> [dest_dir]" >&2
    echo "  known: ${!BACKEND_FILES[*]}" >&2
    exit 1
fi

ENGINE="$1"
FLAVOR="$2"
KEY="${ENGINE}:${FLAVOR}"
DEST_DIR="${3:-backends/${ENGINE}/${FLAVOR}}"

ENTRY="${BACKEND_FILES[$KEY]:-}"
if [ -z "$ENTRY" ]; then
    echo "ERROR: unknown (engine, flavor) '$KEY'. Known: ${!BACKEND_FILES[*]}" >&2
    exit 1
fi

read -r REPO TAG FILENAME EXPECTED_SHA <<< "$ENTRY"
URL="https://github.com/${REPO}/releases/download/${TAG}/${FILENAME}"

mkdir -p "$DEST_DIR"
TMP_TAR="$(mktemp)"
trap 'rm -f "$TMP_TAR"' EXIT

# Skip re-download if a marker from a previous fetch of the SAME sha already
# exists -- same "skip if present and verified" convention fetch_model.sh
# uses, adapted for a directory of extracted files rather than one file.
MARKER="$DEST_DIR/.fetched-sha256"
if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$EXPECTED_SHA" ]; then
    echo "==> $KEY already fetched at $EXPECTED_SHA, skipping"
    exit 0
fi

echo "==> Fetching $KEY ($FILENAME) from $URL"
curl -fL --progress-bar -o "$TMP_TAR" "$URL"

ACTUAL_SHA="$(sha256sum "$TMP_TAR" | cut -d' ' -f1)"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "ERROR: sha256 mismatch for $FILENAME" >&2
    echo "  expected: $EXPECTED_SHA" >&2
    echo "  actual:   $ACTUAL_SHA" >&2
    exit 1
fi

echo "==> Extracting into $DEST_DIR"
# --strip-components=1: every one of these release tarballs wraps its
# contents in one top-level "<name>-<version>-<os>-<arch>/" directory --
# confirmed by actually listing both tarballs' contents while building this.
tar -xzf "$TMP_TAR" -C "$DEST_DIR" --strip-components=1

echo "$EXPECTED_SHA" > "$MARKER"
echo "==> $KEY fetched and verified ($EXPECTED_SHA)"
