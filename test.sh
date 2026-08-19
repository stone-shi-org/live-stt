#!/usr/bin/env bash
#
# Test runner. Emits JUnit XML into test-reports/ for the Atlassian Bamboo
# JUnit Parser task (result pattern: **/test-reports/*.xml).
#
#   ./test.sh                       unit tests only, native, no model/network
#   ./test.sh --docker              canonical path -- avoids the Python 3.14
#                                    grpcio wheel gap on the native venv (see
#                                    CLAUDE.md); image is python:3.12-slim
#   ./test.sh --integration         also run integration tests against the
#                                    real worker binary + a mounted model
#   ./test.sh --slow                also run the long-running suite
#                                    (30-120 min simulated calls)
#   ./test.sh --gpu                 integration tests on the CUDA build
#                                    (requires LSTT_GPU_HOST; Phase 5)
#   ./test.sh -- -k test_framing    extra args after -- go to pytest
#
# Deliberately NOT `set -e`: a failing test run must still leave its XML on
# disk, or Bamboo reports "no test results found" instead of showing which
# test failed.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

REPORT_DIR="$SCRIPT_DIR/test-reports"
VENV_DIR="$SCRIPT_DIR/venv"

RUN_INTEGRATION=0
RUN_SLOW=0
RUN_GPU=0
USE_DOCKER=0
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker)      USE_DOCKER=1; shift ;;
        --integration) RUN_INTEGRATION=1; shift ;;
        --slow)        RUN_SLOW=1; RUN_INTEGRATION=1; shift ;;
        --gpu)         RUN_GPU=1; RUN_INTEGRATION=1; shift ;;
        --)            shift; PYTEST_ARGS=("$@"); break ;;
        -h|--help)     sed -n '2,17p' "$0"; exit 0 ;;
        *)             PYTEST_ARGS+=("$1"); shift ;;
    esac
done

# Bamboo fails a build when result files predate the build start, so a stale
# XML from a previous run is worse than no XML at all. Always start clean.
rm -rf "$REPORT_DIR"
mkdir -p "$REPORT_DIR"

RC=0

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

MARK_EXPR="not integration"
if [ "$RUN_INTEGRATION" -eq 1 ]; then
    MARK_EXPR="integration"
    [ "$RUN_SLOW" -eq 0 ] && MARK_EXPR="$MARK_EXPR and not slow"
    [ "$RUN_GPU" -eq 0 ] && MARK_EXPR="$MARK_EXPR and not gpu"
fi

# --------------------------------------------------------------------------- #
# Native
# --------------------------------------------------------------------------- #

setup_venv() {
    # Prefer python3.12/3.13 if present on PATH: grpcio's cp314 wheel
    # coverage is recent enough that a fresh checkout on an older host image
    # may hit a slow source build or a hard failure. --docker (python:3.12-
    # slim) is the canonical, always-working path -- this is the fallback.
    local py_bin="python3"
    for candidate in python3.12 python3.13; do
        if command -v "$candidate" >/dev/null 2>&1; then
            py_bin="$candidate"
            break
        fi
    done
    if [ "$py_bin" = "python3" ]; then
        echo "--> WARNING: python3.12/3.13 not found on PATH; falling back to" \
             "$(python3 --version 2>&1). grpcio may need to build from source" \
             "or fail outright -- prefer ./test.sh --docker if this hangs."
    fi

    if [ ! -d "$VENV_DIR" ]; then
        echo "--> Creating virtualenv at $VENV_DIR (using $py_bin)"
        "$py_bin" -m venv "$VENV_DIR" || { echo "ERROR: could not create venv" >&2; return 1; }
    fi
    VENV_PYTHON="$VENV_DIR/bin/python"
    [ -x "$VENV_PYTHON" ] || VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
    if [ ! -x "$VENV_PYTHON" ]; then
        echo "ERROR: no python binary inside $VENV_DIR" >&2
        return 1
    fi
    echo "--> Installing dependencies"
    "$VENV_PYTHON" -m pip install --upgrade pip --quiet
    "$VENV_PYTHON" -m pip install -r requirements.txt --quiet
}

run_native() {
    banner "pytest -m \"$MARK_EXPR\""
    setup_venv || return 1
    if [ ! -d "live_stt/pb" ]; then
        echo "--> Generating protobuf stubs"
        bash scripts/gen_proto.sh || return 1
    fi
    "$VENV_PYTHON" -m pytest tests/ -v \
        -m "$MARK_EXPR" \
        --junitxml="$REPORT_DIR/unit-results.xml" \
        --junit-prefix=unit \
        "${PYTEST_ARGS[@]}"
}

# --------------------------------------------------------------------------- #
# Docker -- for CI agents without a matching Python toolchain, or to reliably
# reproduce the offline-safety guarantee (test-unit ships no libparakeet.so
# and no model at all, so an accidental integration-shaped unit test would
# fail the build instead of silently passing against the real thing).
# --------------------------------------------------------------------------- #

run_docker() {
    banner "pytest -m \"$MARK_EXPR\" (in Docker)"
    if [ "$RUN_INTEGRATION" -eq 1 ]; then
        if [ ! -d "$SCRIPT_DIR/models" ]; then
            echo "ERROR: --integration --docker needs ./models mounted (scripts/fetch_model.sh)" >&2
            return 1
        fi
        local target="test-integration"
        [ "$RUN_GPU" -eq 1 ] && target="test-integration-cuda"
        docker build --target "$target" -t "live-stt-$target" . || return 1
        docker run --rm \
            -v "$REPORT_DIR:/app/test-reports" \
            -v "$SCRIPT_DIR/models:/models:ro" \
            "live-stt-$target"
    else
        docker build --target test-unit -t live-stt-test-unit . || return 1
        docker run --rm \
            -v "$REPORT_DIR:/app/test-reports" \
            live-stt-test-unit
    fi
}

# --------------------------------------------------------------------------- #

if [ "$USE_DOCKER" -eq 1 ]; then run_docker; else run_native; fi
RC=$?

banner "Summary"
echo "  tests exit=$RC"
echo ""
echo "  Reports:"
ls -1 "$REPORT_DIR" 2>/dev/null | sed 's/^/    /' || echo "    (none)"
echo ""
echo "  Bamboo: add a JUnit Parser final task with pattern **/test-reports/*.xml"
echo ""

exit "$RC"
