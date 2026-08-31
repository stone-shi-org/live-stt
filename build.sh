#!/usr/bin/env bash
#
# Build and tag the container image.
#
#   ./build.sh                              build live-stt:latest locally, no registry
#   ./build.sh -p                           build and push (requires -r or LSTT_REGISTRY)
#   ./build.sh -t v1.2                      add an extra tag
#   ./build.sh -r registry.example.com/x    tag for a specific registry
#   ./build.sh --cuda                       build runtime-cuda instead, tags get a -cuda suffix
#   ./build.sh --cuda --diarize             build runtime-cuda-diarize (CUDA ASR + pyannote.audio
#                                            baked in, see requirements-diarization.txt), tags get
#                                            a -cuda-diarize suffix -- requires --cuda, no CPU+
#                                            diarize image target exists
#   LSTT_REGISTRY=registry.example.com/x ./build.sh   same, via environment
#
# Registry precedence: -r flag > LSTT_REGISTRY env var > unset (local image only).
# There is deliberately no hardcoded registry default here -- this script is
# checked into a public repo, and baking in a real host would leak it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

REGISTRY="${LSTT_REGISTRY:-}"
IMAGE="live-stt"
PUSH=0
EXTRA_TAG=""
CUDA=0
DIARIZE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--push)     PUSH=1; shift ;;
        -t|--tag)      EXTRA_TAG="$2"; shift 2 ;;
        -r|--registry) REGISTRY="$2"; shift 2 ;;
        --cuda)        CUDA=1; shift ;;
        --diarize)     DIARIZE=1; shift ;;
        -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

log() { echo "==> $*"; }

if [ "$PUSH" -eq 1 ] && [ -z "$REGISTRY" ]; then
    echo "ERROR: --push needs a registry. Pass -r <registry> or set LSTT_REGISTRY." >&2
    exit 1
fi

if [ "$DIARIZE" -eq 1 ] && [ "$CUDA" -ne 1 ]; then
    echo "ERROR: --diarize currently requires --cuda -- no runtime-diarize (CPU) target exists." >&2
    exit 1
fi

# REPO is the full tag prefix: "live-stt" locally, or
# "registry.example.com/homestack/live-stt" once a registry is set.
REPO="${REGISTRY:+${REGISTRY}/}${IMAGE}"

TARGET="runtime"
SUFFIX=""
if [ "$CUDA" -eq 1 ]; then
    TARGET="runtime-cuda"
    SUFFIX="-cuda"
fi
if [ "$DIARIZE" -eq 1 ]; then
    TARGET="runtime-cuda-diarize"
    SUFFIX="-cuda-diarize"
fi

# version.txt is baked into the image and surfaced at GetServerInfo (and
# /api/version), so a running container can always be traced back to a
# commit. git status --porcelain, not a two-way diff --quiet check, so an
# untracked file counts as dirty too. parakeet_ref/whisper_ref are the
# upstream submodule SHAs -- PARAKEET_VERSION is frozen at "0.0.1" upstream
# (whisper.cpp does carry a real version, 1.9.3-dev as of this pin, but the
# SHA is still recorded for the same reason: it is the only thing that
# identifies the EXACT vendored commit, not just a release line).
FULL_HASH="$(git rev-parse HEAD 2>/dev/null || true)"
TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PARAKEET_REF="$(git -C worker/third_party/parakeet.cpp rev-parse HEAD 2>/dev/null || echo unknown)"
WHISPER_REF="$(git -C worker/third_party/whisper.cpp rev-parse HEAD 2>/dev/null || echo unknown)"

if [ -z "$FULL_HASH" ]; then
    HASH="dev"
else
    HASH="${FULL_HASH:0:7}"
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        HASH="${HASH}-dev"
    fi
fi

{
    echo "hash=${HASH}"
    echo "timestamp=${TIMESTAMP}"
    echo "parakeet_ref=${PARAKEET_REF}"
    echo "whisper_ref=${WHISPER_REF}"
} > version.txt

log "Version: ${HASH} (${TIMESTAMP}), parakeet.cpp @ ${PARAKEET_REF:0:7}, whisper.cpp @ ${WHISPER_REF:0:7}"

log "Building ${REPO}:latest${SUFFIX} (target ${TARGET})"
docker build --target "$TARGET" \
    -t "${REPO}:latest${SUFFIX}" \
    -t "${REPO}:${HASH}${SUFFIX}" \
    ${EXTRA_TAG:+-t "${REPO}:${EXTRA_TAG}${SUFFIX}"} \
    .

log "Built:"
docker images "${REPO}" --format '    {{.Repository}}:{{.Tag}}  {{.Size}}'

if [ "$PUSH" -eq 1 ]; then
    log "Pushing"
    docker push "${REPO}:latest${SUFFIX}"
    docker push "${REPO}:${HASH}${SUFFIX}"
    [ -n "$EXTRA_TAG" ] && docker push "${REPO}:${EXTRA_TAG}${SUFFIX}"
    log "Pushed"
elif [ -z "$REGISTRY" ]; then
    log "Built locally as ${REPO}:latest${SUFFIX}. Set -r/LSTT_REGISTRY and pass -p to push."
else
    log "Not pushed. Re-run with -p to push to ${REGISTRY}."
fi
