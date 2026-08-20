# CLAUDE.md

Streaming ASR service wrapping [parakeet.cpp](https://github.com/mudler/parakeet.cpp)'s streaming
C API, exposed over bidirectional gRPC to a Python telephony application:

```
one phone call = one gRPC Transcribe stream = one logical ASR session
```

## Serious open risk — read this before trusting rotation or crash recovery

**Starting a fresh parakeet.cpp stream at certain audio content can cause it to silently drop
several SECONDS of clear, present speech partway through the stream — with no error, no warning,
just missing words. Found while testing Phase 3's rotation end to end against the real model, and
confirmed to be an upstream/engine behavior, not a bug in this codebase's rotation logic.**

**Reproduction** (`tools/repro_stream_start_dropout.py`, verified working as of this writing):
against `~/src/transcript/output.wav` (16kHz mono, verified fixture), a **single worker process
with zero rotation/dual-feed/promotion code involved** — just `parakeet_capi_stream_begin` fed
audio starting at file offset **8.00s** — never transcribes "yes yes absolutely ok perfect well
yeah" (7 words, ~3.6s of unambiguous speech that occurs ~10 seconds into that stream, at absolute
file position ~18.08-21.68s). The exact same worker, model, and code, started at file offset
**7.84s or 8.16s instead — each only 160ms away** — transcribes those same words correctly. Six
offsets 160ms apart were probed; only the one landing exactly on 8.00s (= 50 model chunks = 100
encoder frames from file start, suspiciously round numbers) failed:

| start offset | "yes yes absolutely..." transcribed? |
|---|---|
| 7.68s | yes |
| 7.84s | yes |
| **8.00s** | **NO — silently dropped** |
| 8.16s | yes |
| 8.32s | yes |
| 8.48s | yes |

This was discovered because a rotation test showed a duplicated/missing word at a cutover seam.
Investigation ruled out every piece of this codebase's own logic, in order: (1) the dedup slack
window, (2) EOU-triggered vs deadline-only cutover (forced deadline-only, bug persisted), (3) the
shadow-worker dual-feed/promotion mechanism itself (a hand-rolled two-process dual-feed-then-kill
reproduction, bypassing `session.py` entirely, showed the SAME drop), (4) finally isolated to a
**single worker, no dual-feed, no promotion at all** — just sensitive to which exact sample its
own stream happens to start on. See git history around this entry for the elimination sequence if
reproducing the investigation.

**Why this matters more than it might first appear:** it is not a resource/timing problem with a
graceful degradation path (like the #63 leak) — it is a **silent correctness failure**. The engine
returns well-formed, HTTP-200-shaped JSON either way; there is no error to catch, no status to
check, no signal that anything went wrong. Both **rotation** (a shadow worker begins a fresh
stream at whatever sample happens to be current when a threshold trips) and **crash recovery** (a
replacement worker begins a fresh stream at whatever sample was live when the crash happened)
create a brand-new stream at an **effectively arbitrary, uncontrolled audio offset** relative to
the call's content — exactly the condition this bug needs. Neither the dedup logic nor any check
currently in this codebase would catch it if it fires in production: the transcript would just be
missing words, indistinguishable from the model simply mis-hearing something.

**Frequency, characterized** (`tools/sweep_stream_start_dropout.py`, results saved at
`tools/sweep_results_example.csv`): built one continuous 82s baseline transcript with word
timestamps, then swept 376 fresh-stream start offsets 160ms apart across the first 62s of the same
file, each fed a 20s window, scoring word recall against the baseline in the matching absolute-time
range (3s startup grace, 0.75s timing tolerance). **3 of 376 offsets (0.8%) scored recall < 0.6**
(0.406, 0.542, 0.558), with a wider band of offsets around 17.4–18.0s scoring a consistently
moderate ~0.61–0.63 — a softer version of the same failure, not just sweep noise. Spot-checked the
worst (offset 29.68s, recall 0.406) word-by-word against the baseline: the candidate cleanly
**drops "twenty thirty minutes or so and i just wanted to informally contact to chat a little bit
more about what's going on here at reddit" in its entirety** (~10s, 15 words), then resumes and
matches the baseline closely from "our business as well as..." onward — the exact same
clean-block-of-silence shape as the original single-offset repro, not a diffuse mismatch.

**The specific bad offsets are NOT fixed properties of the audio alone — they shift with
`n_threads`.** The original repro (8.00s failing, 7.84s/8.16s fine) used `n_threads=4`; this sweep
used `n_threads=1` throughout and did not flag 8.00s at all (nearby grid points 7.92s/8.08s both
scored ~0.96), instead flagging different nearby offsets (8.72s among them). Since ggml's threaded
reduction order changes floating-point summation order, and floating-point addition is not
associative, this points to a **genuine numerical near-tie in some incremental decoding decision**
(plausibly the RNN-T joint network's blank/non-blank gate, or a chunked-attention boundary weight)
that a few ULPs of difference can tip either way — and tipping it the "wrong" way apparently drives
the decoder into a state it takes several seconds to recover from. This also means the specific
offsets found here are not portable to a different `n_threads`, a different quantization, a
different CPU (SIMD reduction order), or a GPU build — only the *rate* (very roughly ~1% of
arbitrary restart points, on this one file) is likely to be a useful estimate, and even that should
not be assumed to generalize without testing on more content.

**Status: unresolved, not yet reported upstream, not mitigated in code.** The rotation state
machine itself (triggers, budget/reserve accounting, dual-feed bookkeeping, EOU/deadline cutover,
crash recovery, seam dedup) is fully implemented and passes 6/6 tests against a controllable fake
worker (`tests/test_rotation.py`) that exercise every structural path correctly. What is NOT
proven safe is what happens when a REAL rotation or crash-recovery lands a fresh stream on a "bad"
starting offset in production — and at a measured ~1% incidence, this is not a theoretical edge
case: a service doing dozens of rotations a day should expect to hit it. Do not treat Phase 3 as
production-ready until this is either understood, reported and fixed upstream, or mitigated
(candidate mitigations, none implemented: report this pattern to `mudler/parakeet.cpp` with the
minimal repro and the sweep methodology above, since the numerical-near-tie theory is exactly the
kind of thing a maintainer with access to the RNN-T joint network's internals could confirm or
rule out quickly; extend the sweep to more/longer content and other `n_threads` values to check
whether ~1% holds up; if it does, consider whether a rotation/crash-recovery could detect the
failure signature live — e.g. an unusually long stretch with zero finalized words despite
non-silent input — and retry with a fresh worker rather than accepting the loss silently, though
that itself needs an audio-activity signal this codebase does not yet compute).

## The one constraint that shapes everything

**parakeet.cpp issue [#63](https://github.com/mudler/parakeet.cpp/issues/63) — open, unfixed.**
`parakeet_capi_stream_feed`/`_json` leaks memory linearly with audio fed, and `stream_free` +
`stream_begin` does **not** reclaim it — only killing the process does. The issue reports
19–41 MB **per second of audio fed**, measured on a CUDA Jetson build, and was never tested on CPU.

**Measured on this repo's own CPU build (`tools/leak_curve.py`, Phase 1 Gate A — 600s runs, both
conditions, linear regression): silence 0.076 MB/s (R²=0.69, oscillating rather than monotone),
real speech 0.085 MB/s (R²=0.87, smoother). Both roughly 0.08 MB/s — 200–500x BELOW the CUDA
number.** This is a rigorous measurement (not a spot check — two 600-second runs, R² checked),
repeatable via `tests/test_leak_curve.py`'s tripwire (shorter 90s runs asserting the same order of
magnitude). At this rate a 2-hour call leaks on the order of tens of MB, not tens of GB — the
worker-rotation machinery below is a real safety net worth keeping, but `rotate_after_sec` is sized
generously (3600s) rather than defensively, because the CPU backend plainly isn't reproducing the
severity of the upstream report. **This does NOT transfer to a CUDA build — re-run Gate A on both
`VmRSS` and per-process VRAM before Phase 5, and don't assume the same headroom.**

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
  `GetServerInfo` are real and working.
- **Phase 1 (all three gates run for real, see "Phase 1 measurements" below):** the leak curve
  (both conditions, 600s each), the thread sweep (n_threads 1-6 on this 6-core host), and the
  telephony-band WER penalty (one NOTSOFAR-1 meeting, three arms). All three have permanent
  regression tests (`tests/test_leak_curve.py`, `tests/test_telephony_band.py`) and standalone
  re-runnable tools (`tools/leak_curve.py`, `tools/thread_sweep.py`, `tools/telephony_band_wer.py`).
- **Phase 2 (proven end-to-end, real gRPC client through a real worker and model — see
  `scripts/e2e_grpc_smoke.py`):** `live_stt/worker.py` (the production `WorkerHandle`, asyncio
  streams over the same socketpair IPC, the `pass_fds`-plus-`preexec_fn` fix baked in),
  `live_stt/session.py` (`CallSession`: `start`/`feed_audio`/`finalize`/`close`, chunk coalescing
  to `model_chunk_ms` — verified by feeding 20ms client frames against a 160ms model chunk and
  confirming `audio_offset_sec` only advances once a full chunk is actually fed), and
  `servicer.Transcribe` wired to it for real, including config validation (encoding, sample rate,
  unknown model), half-close → `finalize()` → `Final`, and a minimal immediate-reject
  `RESOURCE_EXHAUSTED` admission counter (NOT the full reserve/gate/trailers design in "Concurrency
  and capacity" below — that's still Phase 3). `tests/fakes/fake_worker_main.py` is a real
  subprocess speaking the real IPC protocol, driven by env vars (crash/abort/hang/leak/rtf/words),
  making `tests/test_session.py` and `tests/test_servicer.py` (a real `grpc.aio` server on a
  loopback port) fast and offline while still exercising real process/fd/signal behavior. No
  rotation yet (Phase 3): one `WorkerHandle` lives for the whole call.
- **Phase 4 (proven end-to-end via a real `docker compose up` against the real model, active call
  mid-drain included):** `live_stt/metrics.py` (`prometheus_client`, exposed at `GET /metrics`),
  `live_stt/redaction.py` (the two-switch `LSTT_TRANSCRIPT_LOG`/`LSTT_AUDIO_DUMP` design, validated
  at startup — refuses to start rather than silently downgrading), `live_stt/state.py`
  (`ServerState`: the one shared object between `server.py`/`servicer.py`/`admin_http.py` — flags
  and the `WorkerBudget`, deliberately NOT a session registry), `live_stt/admin_http.py`
  (`/api/health` now capacity- and draining-aware, `/api/stats`, `/metrics`), and a real SIGTERM
  handler in `server.py` that flips gRPC health to `NOT_SERVING` and `state.draining = True`
  **immediately**, before `grpc.aio`'s own drain grace period even starts. Verified manually: with
  one call in flight, SIGTERM makes `/api/health` report `"draining"` and `/api/stats` report
  `active_calls: 1` right away, a *second* call attempted during that window gets
  `UNAVAILABLE`, and the *original* in-flight call keeps running and reaches a normal `Final` —
  confirmed via the lifecycle log line, which also confirms redaction is doing its job by showing
  `chars=380 words=75` with **no transcript text**, the default (`off`) mode. `docker-compose.yml`
  built and run for real (`docker compose up -d --build`) — not just `docker build`. Along the
  way, running real load in a container (not just natively) surfaced a genuine grpc-core
  fork-safety crash affecting every worker spawn in the service, not just tests — see
  "Concurrency and capacity" below for the finding and the `GRPC_POLL_STRATEGY=poll` fix.
  `docker-compose.gpu.yml` intentionally not written yet: there is no `runtime-cuda` Dockerfile
  target for it to reference (Phase 5).
- **Docker:** `Dockerfile` has all targets (`parakeet-build`, `worker-build`, `runtime`,
  `test-unit`, `test-integration`, `test-worker`). `test-unit`, `runtime`, and the full
  `docker-compose.yml` stack have all been built and run for real — `runtime`'s worker binary was
  run via `docker exec` against a real volume-mounted model inside the actual container (not just
  natively), confirming the `$ORIGIN` rpath resolves the ggml `.so`s correctly in that environment.
  `test-integration` is written but not yet built+run in a container — verify before trusting it
  blindly.
- **Not yet built:** the AudioRing/backpressure design and its drift watchdog (`queue_max_sec`,
  `ring_history_sec`, `warn_behind_sec`, `abort_behind_sec` exist as `Settings` fields, referenced
  in a docstring, but nothing reads them yet — `feed_audio()` just buffers to one model chunk and
  calls the worker; there is no bounded ring, no drop-oldest policy, and no `behind_sec`
  computation). `LSTT_AUDIO_DUMP` is validated at startup but has nothing to act on (no ring buffer
  to dump from). Sending a live `Warning{SERVER_DRAINING}` event into an *already-open* stream when
  a drain starts is also not implemented — draining currently only blocks *new* admission; an
  active call gets no in-band notice that a deadline is coming, it just has up to
  `drain_timeout_sec` to finish naturally before `grpc.aio`'s own stop-grace forcibly ends it.
  CUDA build, Prometheus scrape-job/Grafana dashboard JSON, long-call and concurrency tests: not
  written yet either.

## Phase 1 measurements (all three gates, run for real on this repo's CPU build)

**Gate A — leak curve.** See the top of this file. `tools/leak_curve.py --condition silence
--audio-sec 600` and `--condition speech --audio-fixture ~/src/transcript/output.wav --audio-sec
600`, both at `n_threads=4`: **0.076 MB/s (silence, R²=0.69) and 0.085 MB/s (speech, R²=0.87)** —
same order of magnitude between conditions, so the leak (what little there is) tracks fed audio
roughly evenly, not emitted tokens specifically. `tests/test_leak_curve.py` pins a wide tripwire
band around this (0.02–0.6 MB/s) as a 90s-run regression check.

**Gate B — thread sweep.** `tools/thread_sweep.py`, 60s of real speech per `n_threads` value, on
this 6-core host:

| n_threads | rtfx | aggregate throughput at W = floor(6×0.8/n_threads) |
|---|---|---|
| 1 | 2.48 | W=4 → **9.92** |
| 2 | 3.87 | W=2 → 7.74 |
| 3 | 4.85 | W=1 → 4.85 |
| 4 | 5.54 | W=1 → 5.54 |
| 6 | 4.75 | W=1 → 4.75 (more threads than cores available for anything else — worse than 4) |

Non-obvious finding: per-thread returns are sharply sub-linear for this model (4x the threads
only bought 2.2x the speed), so **more single-threaded workers beat fewer multi-threaded ones** on
aggregate throughput here — `n_threads_per_worker=1` is the config default. Caveat: this measures
one stream at a time, not real concurrent contention (shared L3/memory bandwidth across
simultaneously-running workers is untested) — a Phase 4 concurrency test on a real multi-core box
is still needed before trusting the W=4 number as a production capacity figure, and this sweep is
dev-host-specific (6 cores) regardless.

**Gate C — telephony band penalty.** `tools/telephony_band_wer.py` against NOTSOFAR-1 meeting
`MTG_32089` (far-field single-channel audio, ~6 min, 1130 reference words):

| arm | WER |
|---|---|
| native (16kHz) | 51.8% |
| linear 8kHz roundtrip (no mu-law) | 67.7% (+15.9 points vs native) |
| mu-law 8kHz roundtrip (full telephony leg) | 70.4% (+18.7 points vs native) |

Narrowbanding costs a real, substantial WER penalty (~30-35% relative), and **most of it is
bandwidth loss, not mu-law's quantization** (mu-law-only cost beyond linear-8kHz: +2.7 points).
Caveat: this is one hard far-field, multi-speaker, single-channel meeting condition — harder than
a real two-party phone call is likely to be, so the *absolute* WER here (51.8% even at native
16kHz) should not be read as "what a phone call will score." The point is the *shape* of the
penalty, not the absolute number, and `tests/test_telephony_band.py` pins the recorded deltas
(with ±0.15 slack) plus the structural claim (`bandwidth_only > codec_only`) as a regression check.

Reproducing Gate C needs `pip install audioop-lts` in the venv (not in `requirements.txt` — it's
only used by `tools/telephony_band_wer.py`'s mu-law *encode* step, which exists purely to simulate
the encode side of a telephony leg for this diagnostic; the service itself never encodes G.711,
only decodes it, so this is deliberately not a runtime dependency of anything under `live_stt/`).

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

**gRPC gotcha found while testing admission rejection** (`tests/test_servicer.py`): if a client
keeps writing request messages on a bidi stream immediately after the server has decided to
`abort()` it (our `RESOURCE_EXHAUSTED` check runs before reading a single message, so this is
easy to hit), the client can occasionally observe a generic `INTERNAL`/`"Internal error from
Core"` instead of the intended status — a client-still-writing-vs-server-aborting race at the
grpc-core transport level, not a servicer bug. A real client that stops writing as soon as it
sees the RPC end (rather than pipelining further messages regardless) won't hit this. Worth
knowing if a future client implementation pipelines aggressively.

**gRPC + fork() crash, found under container load, more severe than the above — mitigated, not
just a test nuisance.** grpc-core's default `epoll1` polling engine has a fork-safety bug:
`WorkerHandle.spawn()`'s `subprocess.Popen` calls `fork()`, and doing that from a process that also
runs a live `grpc.aio.server()` (i.e. **every worker spawn in this entire service** — every call,
every rotation, every crash recovery) can hit `F ... ev_epoll1_linux.cc:1121] Check failed:
next_worker->state == KICKED` and abort the **whole process**, not just the spawn attempt. Reproduced
intermittently (roughly 1-in-8 to 1-in-12 runs) in `tests/test_servicer.py` under
`docker run`, essentially never natively — container CPU scheduling makes the race window easier to
hit, but the hazard is not container-specific; production hits the identical fork pattern on every
single call. `pass_fds` (required for the socketpair fd-passing scheme) unconditionally disables
Python's safer `posix_spawn` path regardless of `preexec_fn`, so this could not be avoided by
trimming what runs in the preexec hook. **Fix, verified 20/20 clean under the same stress that
reproduced it 1-in-8: `GRPC_POLL_STRATEGY=poll`**, set as a Dockerfile `ENV` on the common ancestor
of `runtime`/`test-unit`/`test-integration` (so nothing can run without it) and via
`tests/conftest.py`'s `pytest_configure` for native/ad-hoc runs (must be set before grpc-core's
first channel/server in the process — it picks its polling engine once and caches the choice, so a
per-test fixture is too late). Isolated by testing each independently:
`GRPC_ENABLE_FORK_SUPPORT=1` alone does **not** fix it; switching off `epoll1` is what matters.
Also observed once, not reproduced (5/5 clean retries immediately after): a client's first message
on a freshly-restarted container's first-ever connection got a generic `INTERNAL`/`"Internal error
from Core"` on its own `SendMessageOperation` — plausibly a separate, benign first-connection
warm-up transient, not chased further since a real client already needs to retry on `INTERNAL`
regardless (see the failure semantics elsewhere in this file).

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
1. **Measure before designing the pool — DONE.** `tools/leak_curve.py` (CPU; CUDA still pending,
   Phase 5), `tools/thread_sweep.py`, the telephony-band WER penalty test — all three run for
   real, with recorded numbers and permanent regression tests. See "Phase 1 measurements" above.
2. **Minimal end-to-end, one generation, no rotation — DONE.** `live_stt/session.py`,
   `live_stt/worker.py`, `servicer.Transcribe` wired to a single worker. Proven via a real gRPC
   client through a real worker and model (`scripts/e2e_grpc_smoke.py`) and via
   `tests/test_session.py`/`tests/test_servicer.py` against the fake worker.
3. **Survive the call — implemented, but see "Serious open risk" at the top of this file before
   trusting it.** The overlapped-rotation state machine, `boundary.py` dedup wired in for real,
   reserve-aware admission (`live_stt/admission.py`'s `WorkerBudget`), and crash recovery are all
   built directly into `live_stt/session.py` (no separate `pool/supervisor.py` — the rotation logic
   turned out to belong with the call it rotates, not a standalone pool manager) and pass 6/6 tests
   against the fake worker. What's NOT proven safe: what happens when a real rotation or crash
   recovery lands a fresh stream on a "bad" starting offset — measured at ~1% incidence, not a
   theoretical edge case. The drift watchdog (`behind_sec`, the AudioRing backpressure design) was
   never built — see the "Not yet built" note above.
4. **Production shape — mostly done.** `docker-compose.yml`, `metrics.py`, `redaction.py`, and a
   real graceful-drain SIGTERM handler are built and verified end-to-end (see above). Not done:
   Prometheus scrape-job/Grafana dashboard artifacts (the `/metrics` endpoint itself works and was
   verified to parse; wiring it into the estate's existing `prom/prometheus` instance is a
   deploy-time step, not a code change, and hasn't been done), and in-band `Warning{SERVER_DRAINING}`
   notification to an already-open stream (see above).
5. **GPU and multilingual.** CUDA build target, VRAM-aware admission, nemotron's language-select
   path, a VAD to synthesize turn boundaries for the no-`<EOU>` model.
