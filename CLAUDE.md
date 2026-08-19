# CLAUDE.md

Streaming ASR service wrapping [parakeet.cpp](https://github.com/mudler/parakeet.cpp)'s streaming
C API, exposed over bidirectional gRPC to a Python telephony application:

```
one phone call = one gRPC Transcribe stream = one logical ASR session
```

## The one constraint that shapes everything

**parakeet.cpp issue [#63](https://github.com/mudler/parakeet.cpp/issues/63) — open, unfixed.**
`parakeet_capi_stream_feed`/`_json` leaks memory linearly with audio fed, and `stream_free` +
`stream_begin` does **not** reclaim it — only killing the process does. The issue reports
19–41 MB **per second of audio fed**, measured on a CUDA Jetson build.

**Update, measured on this repo's own CPU build (informal, not yet a rigorous Gate A — see
"What's actually been measured" below): feeding 112s of real speech through one session grew RSS
by only ~26 MB (~0.2 MB/s), two orders of magnitude below the CUDA number.** This is a real,
repeatable local observation, not a citation — but it is far too short a run to trust as a
capacity-planning constant. **Do not size `LSTT_ROTATE_AFTER_SEC` or `LSTT_WORKER_RSS_SOFT_KB`
from it.** Run the full `tools/leak_curve.py` (Phase 1 Gate A: silence AND real speech, to 600s+,
linear regression with R² check) before trusting any number here, on both backends this service
ships. If it holds up, the aggressive worker-rotation design below may turn out to be far more
conservative than the CPU backend actually needs — but the architecture should still support it,
because CUDA (Phase 5) may need it and because "assume the worst until measured" is the only safe
default for a service whose failure mode is OOM on a live phone call.

Given that, the honest invariant this codebase is built on — not "one immortal `parakeet_stream`
per call" — is:

```
one phone call
  = one gRPC Transcribe RPC
  = one logical ASR session
  = an ordered chain of parakeet_stream GENERATIONS, exactly one live at a time
    (two only during a brief, deliberate overlap during rotation)
```

Rotation is cheap in quality terms: `att_context` gives only ~4.5s (120M model) of encoder left
context, and the library already resets the RNN-T decoder LSTM at every `<EOU>` by design — there
is no long-horizon "conversation memory" in a `parakeet_stream` to lose. See the full design
rationale (overlapped dual-feed, seam dedup, rotation triggers) in the implementation plan this
was built from: `~/.claude/plans/write-a-service-that-tranquil-book.md` on the machine this was
designed on — reproduced in condensed form below since that path won't exist on every checkout.

## Architecture

```
telephony app (separate repo)
  RTP/SIP -> jitter buffer -> G.711 mu-law/A-law, 8kHz
  live_stt.client.telephony : LUT decode -> soxr resample -> int16 16kHz mono   <- SHIPPED FROM HERE
  live_stt.client.asr_client: grpc.aio Transcribe(stream)
        |  gRPC / HTTP2, direct on home-net -- NOT through Traefik (see "Ops notes")
        v
+------------------- container: live-stt --------------------------+
| PID 1  live_stt/server.py -- grpc.aio, one event loop            |
|   grpc_health.v1  +  reflection  +  GetServerInfo                |
|   admission.py     asyncio.Semaphore(LSTT_MAX_CONCURRENT_CALLS)  |  <- Phase 2/3, not built yet
|   pool/supervisor.py   warm spares + 1 reserved slot             |  <- Phase 2/3, not built yet
|                                                                   |
|   servicer.Transcribe()      <- one coroutine per call           |  <- currently UNIMPLEMENTED
|     CallSession               <- coroutine-local, NOT in a       |  <- Phase 2/3, not built yet
|       active/incoming WorkerHandle   registry -- no session_id   |
|                                                                   |
|  == socketpair(AF_UNIX) on fd 3, u32-len frames, int16 audio ==  |
|                                                                   |
|   worker (C++)   one process == one call GENERATION              |
|     main.cpp   pk::set_num_threads(N); frame loop                |
|     Session { parakeet_ctx* ctx_; parakeet_stream* stream_; }    |
|       stream_begin_lang() called ONCE, never re-begun            |
|       reports rss_kb + fed_samples on every RESULT/FINAL frame   |
|                                                                   |
|   ./models/*.gguf  mounted read-only, never baked into the image |
+--------------------------------------------------------------------+
```

Where the streaming state lives (one arrow per level, no fan-out, so the invariant above is
literally readable in the code):

```
StreamingASRServicer.Transcribe()   live_stt/servicer.py     (currently a stub -- see below)
  -> CallSession                    live_stt/session.py      (Phase 2/3, not built yet)
       -> WorkerHandle active       live_stt/worker.py        (Phase 2/3, not built yet)
            -> worker process
                 -> Session         worker/session.hpp
                      |- parakeet_ctx*    ctx_
                      \- parakeet_stream* stream_
```

`worker/session.hpp`'s `Session` is the actual bottom of that chain today, and it enforces the
anti-issue-#13 invariant ("never call stream_begin twice") by construction: `stream_` is set once
in `configure()` and there is no loop anywhere near it.

## What's actually been implemented (read this before assuming a phase is done)

- **Phase 0 (proven end-to-end):** the C++ worker builds, loads the 120M EOU model, runs a real
  streaming session over real speech, produces a sensible multi-word transcript, and exits
  cleanly. `tests/test_capi_smoke.py` encodes this as a repeatable, real-binary integration test
  (6/6 passing as of this writing). The gRPC server, health/reflection services, and
  `GetServerInfo` are real and working; `Transcribe` returns `UNIMPLEMENTED` on purpose.
- **Not yet built (Phase 2/3):** `live_stt/session.py` (CallSession, the 4-coroutine structure,
  AudioRing/backpressure, the rotation state machine), `live_stt/worker.py` (WorkerHandle:
  spawn/socketpair/rlimits — the actual production analogue of `tests/worker_harness.py`),
  `live_stt/pool/` (spares, admission), `live_stt/boundary.py`'s dedup wired into a real rotation,
  `live_stt/redaction.py`, `live_stt/metrics.py`. `Transcribe` is a stub.
- **Not yet run:** the Phase 1 measurement gates as rigorous, decision-grade artifacts
  (`tools/leak_curve.py`, `tools/thread_sweep.py`, the telephony-band WER penalty test). Only an
  informal, short leak spot-check has been done (see above).
- **Docker:** `Dockerfile` has all targets (`parakeet-build`, `worker-build`, `runtime`,
  `test-unit`, `test-integration`, `test-worker`) and `test-unit` has been built and verified.
  `runtime`/`test-integration` are written but not yet built+run end-to-end in a container as of
  this writing — verify before trusting them blindly.
- **docker-compose.yml, CUDA build, observability (metrics/redaction), long-call and concurrency
  tests:** not written yet.

## Model choice

Ships both, default `realtime_eou_120m-v1`:

| | `realtime_eou_120m-v1` (default) | `nemotron-3.5-asr-streaming-0.6b` |
|---|---|---|
| Languages | English only | 40 locales (19 production-ready) |
| `<EOU>`/`<EOB>` | **Yes** — real vocab entries, p50 ~160ms | **No** — its special tokens are language tags (`<en-US>`, ...); parakeet.cpp resolves EOU by vocab lookup so `eou_id_` stays -1 and `*eou_out`/`"eou"` is always 0. **Do not build turn detection on it.** |
| Punctuation/caps | No | Yes |
| Chunk size | 160ms (baked into the GGUF) | 320ms (baked into the GGUF) |
| Quirk | — | Leaks a language tag into `text`, e.g. `"...eyes. <en-US> It is..."` — stripped by `live_stt/models.py::strip_language_tag` when `strip_language_tag=True` |

The EOU model is the default because its real `<EOU>` makes the worker-rotation cut point exact
(cut at the boundary, no fuzzy text alignment) and it's ~5x cheaper per stream. `StreamConfig.model`
/ `.language` select nemotron for multilingual calls.

## parakeet.cpp: verified API facts and build gotchas found while building this

The submodule is pinned at a specific SHA (`.gitmodules` / `git -C worker/third_party/parakeet.cpp
rev-parse HEAD`, also baked into `version.txt` by `build.sh` as `parakeet_ref`). Treat the
following as true **at that pin**, not as permanent facts about the project:

- **`parakeet_capi.h`** (`include/`) is the real API: `parakeet_capi_load` / `_free`,
  `parakeet_capi_stream_begin[_lang]`, `parakeet_capi_stream_feed[_json]`,
  `parakeet_capi_stream_finalize[_json]`, `parakeet_capi_stream_free`, `parakeet_capi_last_error`.
  ABI v6 at the pinned SHA. `stream_feed_json`/`stream_finalize_json` are what this worker uses —
  never mix with the typed `drain_events` path on the same stream, they share one event queue.
- **`pk::set_num_threads(int)` and `pk::shutdown_backend()` have NO C-API equivalent**, and live in
  a **private** header: `src/ggml_graph.hpp`, not `include/`. This worker reaches into it anyway
  (`worker/main.cpp`'s `#include "ggml_graph.hpp"`, with an extra `target_include_directories`
  pointed at `parakeet.cpp/src` in `worker/CMakeLists.txt`) — this is **the decisive reason the
  worker is C++ and not a ctypes/purego binding**: those symbols are unreachable any other way.
  This is explicitly not part of upstream's public contract; a future pin bump that moves or
  removes this header breaks the build here, which is the point — a loud failure, not a silent one.
- **`parakeet.cpp`'s own `CMakeLists.txt` is not `add_subdirectory()`-safe.** It references
  `${CMAKE_SOURCE_DIR}/third_party` (for `dr_wav.h`) and
  `${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh`, both assuming it is the top-level CMake
  project. `CMAKE_SOURCE_DIR` always resolves to the **outermost** project, so nesting it via
  `add_subdirectory` from another project (which is what a naive `worker/CMakeLists.txt` would do)
  **silently breaks the `dr_wav.h` include path** (a hard build failure) **and silently skips the
  ggml-patch step** (the `EXISTS` guard on the patch script just resolves to nothing and no-ops,
  with zero warning). Discovered by actually trying it, not by reading the docs. **Fix: build it
  standalone** (`scripts/build_worker.sh`'s first `cmake -S third_party/parakeet.cpp -B
  build-parakeet ...`), then have `worker/CMakeLists.txt` link against the resulting
  `libparakeet.a` as an `IMPORTED` target plus the ggml `.so`s, rather than nesting it.
- **`ggml` builds as shared libraries even when `PARAKEET_SHARED=OFF`.** `PARAKEET_SHARED` only
  controls whether `libparakeet.a`/`.so` itself is static or shared; ggml's own backend-registry
  architecture (`ggml-backend-reg.cpp`/`ggml-backend-dl.cpp`) builds `libggml.so`,
  `libggml-base.so`, `libggml-cpu.so` (and `libggml-cuda.so` on a CUDA build) regardless. The
  worker executable is statically linked against `libparakeet.a` but **dynamically** against these
  three (four with CUDA) — they must ship alongside the binary. `worker/CMakeLists.txt` sets
  `INSTALL_RPATH "$ORIGIN"` so the binary finds them **co-located in the same directory**, no
  `LD_LIBRARY_PATH` or `ldconfig` needed — the Dockerfile's `runtime` stage copies them there for
  exactly this reason. Locally (not in Docker), you still need `LD_LIBRARY_PATH` pointed at
  `worker/build-parakeet/third_party/ggml/src` since the binary isn't in that directory during
  native `./scripts/build_worker.sh` development — see `tests/worker_harness.py`.
- **`GGML_NATIVE=OFF` alone silently ships a scalar build.** It correctly disables `-march=native`
  (needed for a portable image — the dev host that builds it is not the deployment target), but
  without *also* setting `GGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON` explicitly, you lose 2-4x
  throughput with **zero visible symptom other than a mediocre RTF months later**. Verified this
  is set correctly at build time (`cmake` configure output lists the exact compiler flags per
  backend variant) and at runtime — see next point.
- **Checking ISA features at runtime: use ggml's real API, not preprocessor macros.** The first
  version of `ggml_feature_string()` in `main.cpp` checked `__AVX2__`/`__FMA__`/`__F16C__` — which
  only reflects how *that translation unit* (`main.cpp`) was compiled, not how ggml-cpu's sources
  were (they get per-file `-mavx2` etc from ggml's own CMake, main.cpp does not). This reported
  `"scalar"` for a build that was, provably, AVX2 — caught by actually running it. Fixed by calling
  `ggml_cpu_has_avx2()`/`_fma()`/`_f16c()` from the public `<ggml-cpu.h>`, which query the actually
  loaded backend. `Ready.ggml_features` and `ServerInfoResponse.ggml_cpu_features` exist
  specifically so this class of bug is visible in one request instead of six months of bad RTF —
  don't regress the check back to compile-time macros.
- **`parakeet-cli --stream` is not a model to copy for real streaming.** It calls
  `pk::run_stream_over_pcm()`, which processes a whole clip in one pass (`src/streaming.hpp`
  documents this explicitly) — a simulation over a complete file, not incremental feeding. Only
  the C API `parakeet_capi_stream_feed[_json]` is genuinely incremental; this worker uses that
  exclusively.
- **`ctx->last_error` is an unsynchronized `std::string` mutation** inside the library
  (`parakeet_capi_stream_feed` clears/sets it outside any lock). Irrelevant here by construction:
  one worker process holds exactly one `ctx` and one thread ever touches it, so there is no
  concurrent access to race on. If a future change ever puts two sessions in one process (it
  shouldn't — see "Concurrency" below), this reappears and needs a mutex around every individual C
  call, held across the call and the subsequent `last_error` read, never across a session's
  lifetime (this is the pattern LocalAI's Go backend uses for the same reason).
- **Global compute mutex**: `src/ggml_graph.cpp` holds a process-global `std::mutex
  g_backend_mutex` for the *entire* `run_graph()` compute — all inference in a process is
  serialized. This is the reason concurrency scales by **process**, not thread: one worker per
  call, full stop, not "a few sessions per worker to save memory."

## IPC protocol (worker <-> front door)

Fixed-fd `socketpair(AF_UNIX, SOCK_STREAM)` on fd 3. Wire shape and frame types are defined once,
in two places that must stay in sync: `live_stt/framing.py` (Python) and `worker/framing.hpp`/`.cpp`
(C++). See either docstring for the byte layout (`u32 length_le | u8 type | payload`).

**Subprocess fd-passing gotcha, confirmed empirically while building this (see git history around
`tests/worker_harness.py` / `scripts/smoke_worker.py`):** to hand the worker a fixed fd 3, you
`os.dup2(child_sock.fileno(), 3)` inside `preexec_fn`. That is necessary but **not sufficient** —
Python's `subprocess.Popen` close-fds pass runs **after** `preexec_fn`, and it only spares fds
listed in `pass_fds`. A dup2 target created *inside* `preexec_fn` is not automatically exempted, so
if `3` isn't *also* listed in `pass_fds`, the close-fds pass immediately closes the very fd you just
created, and the worker sees `EBADF` on fd 3 (observed as an immediate `ConnectionResetError` on the
parent's next `recv`, or the worker exiting with code 1 having never blocked on the CONFIG read at
all). **Fix:** `pass_fds=(child_sock.fileno(), 3)` — list both the real fd and the target number.
This bit both `scripts/smoke_worker.py` and the debug session that found it; the real
`live_stt/worker.py` (Phase 2) must do this correctly too, and should have a unit test pinning it
(a fake worker script that just echoes what fd it received on, run via the real spawn path).

## Audio boundary

Client-side conversion, one canonical wire format. The service accepts **16kHz mono int16 LE
only**; `live_stt/client/telephony.py` (µ-law/A-law LUT decode + `soxr` 8k→16k resample) ships from
this repo so telephony apps import it rather than reinventing it. LUTs are verified byte-for-byte
against `audioop-lts`'s reference implementation for all 256 codes (`tests/test_telephony.py`,
golden vectors in `tests/fixtures/{ulaw,alaw}_lut_golden.json`) — `audioop` itself was removed by
PEP 594 in Python 3.13 (this host runs 3.14; `import audioop` fails), which is why the LUT approach
exists at all rather than depending on it.

int16→float32 conversion happens **exactly once**, in `worker/pcm.hpp`, immediately before
`stream_feed_json` — float32 never appears on any wire in this service.

## Concurrency and capacity

One worker process per call generation, unconditionally — not "a few sessions per worker." This
single decision makes four separate hazards structurally impossible at once: the `last_error` race
(one thread, one ctx), the global-mutex ceiling (never contended), a CUDA `abort()`'s blast radius
(one call), and the #63 leak (bounded by, and fully reclaimed by, killing one call's worth of
state). Target is 3-8 concurrent calls per instance — needs a ~20-core box; **this repo's dev host
is 6 cores and sustains ~1-2 concurrent streams, functional testing only, never a capacity signal.**

## Conventions

House conventions, mirrored from `~/src/my-meeting-notes`: flat top-level package (no `src/`
layout), `requirements.txt` with a reason comment on every non-obvious pin, `pyproject.toml` holds
only `[tool.pytest.ini_options]`, stdlib `logging` + `dictConfig` under a namespaced
non-propagating root logger (`stt`, not `mmn`), Python-urllib `HEALTHCHECK` (no curl in the image),
`build.sh`/`test.sh` with the same flag shapes, `version.txt` baked at build time (`hash`,
`timestamp`, plus `parakeet_ref` — the upstream SHA, since `PARAKEET_VERSION` is frozen at `0.0.1`
upstream and the SHA is the only real version signal for the vendored engine).

Protobuf stubs are **generated, not checked in** (`live_stt/pb/`, gitignored;
`scripts/gen_proto.sh` regenerates, invoked by `test.sh`/`build.sh`/the Dockerfile). The generated
`*_grpc.py` imports its sibling `pb2` module via the proto package path
(`from livestt.v1 import asr_pb2`), not a relative import — `gen_proto.sh` handles this by writing
a `sys.path`-inserting `__init__.py` and touching `__init__.py` in the generated package dirs; see
that script's comment if this ever breaks.

Tests: flat `tests/test_*.py` + `conftest.py` + `fixtures/`, `pytest` markers `integration`
(house-standard, verbatim), `model` (needs `LSTT_MODEL_PATH`/`./models`), `slow`, `gpu`. Unit tests
must never touch the network, a real model, or the worker binary — `tests/conftest.py`'s autouse
fixture enforces this by pointing model-ish env vars at nonexistent paths, and the Docker
`test-unit` target ships no worker binary and no model at all, so an accidentally-integration-shaped
"unit" test fails the build loudly instead of silently passing against the real thing.
`tests/worker_harness.py` / `scripts/smoke_worker.py` are the reusable pieces for driving the real
binary from Python in integration tests — see the fd-passing gotcha above before touching either.

## Ops notes

- **Do not route gRPC through Traefik on `web-secure`.** (Not yet verified against *this* host's
  `traefik-proxy` config in this session — carried over from the design phase; re-check
  `traefik-proxy/docker-compose.yml`'s `respondingTimeouts` before wiring up compose.) If it still
  matches what was found during design, `writeTimeout=600s` on that entrypoint would silently
  truncate every call at 10 minutes; `readTimeout=0`/`idleTimeout=0` is not sufficient on its own.
  Dial the service directly on `home-net`; put Traefik on the admin/metrics port only, or add a
  dedicated gRPC entrypoint with all three timeouts at 0 plus `loadbalancer.server.scheme=h2c`.
- **`home-net` (hyphen) is the real docker network on this host**, not `home_net` (underscore, which
  appears in one bootstrap script but doesn't exist as an actual network here).
- Phase 5 (CUDA) targets are written into the Dockerfile as scaffolding
  (`parakeet-build-cuda`/`runtime-cuda` are NOT yet split out — only CPU targets exist right now)
  but are unbuilt and unverified; this dev host has no nvidia runtime at all, so that path can only
  be smoke-tested on the GPU box.

## Implementation phases (see the original plan for full detail)

0. **Skeleton + first light — DONE.** Worker builds and runs against a real model; gRPC/health/
   reflection/GetServerInfo work; `test-unit` Docker target builds.
1. **Measure before designing the pool — NOT DONE AS RIGOROUS GATES.** `tools/leak_curve.py`
   (CPU **and** CUDA), `tools/thread_sweep.py`, the telephony-band WER penalty test. Only an
   informal leak spot-check exists so far (see top of this file). No pool config should be
   finalized before these run properly.
2. **Minimal end-to-end, one generation, no rotation.** `live_stt/session.py`,
   `live_stt/worker.py` (the production `WorkerHandle`, mind the fd-passing gotcha),
   `servicer.Transcribe` wired to a single worker, no rotation yet.
3. **Survive the call.** `live_stt/pool/supervisor.py`, the overlapped-rotation state machine,
   `boundary.py` wired in for real, admission control, the drift watchdog.
4. **Production shape.** `docker-compose.yml`, `metrics.py`, `redaction.py`, graceful drain,
   Prometheus/Grafana wiring.
5. **GPU and multilingual.** CUDA build target, VRAM-aware admission, nemotron's language-select
   path, a VAD to synthesize turn boundaries for the no-`<EOU>` model.
