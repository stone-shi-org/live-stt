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

*One data point consistent with the above, added during Phase 5 (see "GPU and multilingual"
below): the original failing configuration — offset 8.00s, `n_threads=4`, same file — was re-run on
the CUDA backend on the real RTX 3090 and did **NOT** drop the words ("yes yes absolutely okay
perfect well yeah" transcribed correctly). That is exactly what the backend-specific-reduction-order
theory predicts (CUDA's parallel reduction order differs entirely from CPU's threaded one), so it
**supports** that theory and **does not** mean CUDA is immune — no offset sweep has been run on
CUDA at all, so the ~1% rate is unmeasured on that backend and must be assumed to still apply, at
different offsets. Nothing here changes the status below.*

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
severity of the upstream report. **None of this transfers to a CUDA build — which is why Gate A was
re-run on both `VmRSS` and per-process VRAM in Phase 5 rather than assuming the same headroom:**

**Gate A re-run on CUDA (Phase 5, real RTX 3090, 300s of real speech, `n_threads=4`,
`tools/leak_curve.py` now sampling per-process VRAM via `nvidia-smi --query-compute-apps` alongside
`VmRSS`): 11.95 MB/s RSS (R²=0.76) and 2.50 MB/s VRAM (R²=0.49) as raw slopes, but
`plateau_detected: true` — and the plateau is the real story.** RSS went 415 MB → ~4.37 GB by
roughly the 150s mark then stayed flat (4373–4386 MB) for the remaining 150s; VRAM went 474 MB →
1828 MB over the same first half then sat at **exactly 1828 MB** for the whole second half. That
shape — early ramp then genuine flat — is consistent with one-time buffer-pool/workspace allocation
sized to the largest shapes seen plus CUDA context and graph-capture overhead, and is
*structurally* different from the CPU backend's near-fully-linear curve (speech, R²=0.87, no
plateau). So: real, moderately strong evidence that #63's per-audio-second leak either does not
manifest the same way on CUDA or is too small to separate from ramp-up in a 300s window — but
**not a settled conclusion.** A 600s run (matching the CPU methodology) is needed to show the
plateau holds over hours rather than minutes before trusting long CUDA calls with no rotation
safety net, and the raw CSV from this run was never promoted into the repo (see Phase 5 below).

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
  `docker-compose.gpu.yml` now exists (Phase 5) as an override layered over `docker-compose.yml`,
  referencing the real `runtime-cuda` target.
- **Phase 5 (CUDA build chain, real-GPU transcription, Gate A on CUDA, and the nemotron path all
  verified for real on 10.100.0.50 — see "Ops notes" for the three build bugs found doing it):**
  `docker build --target runtime-cuda -t live-stt:latest-cuda .` succeeds cleanly from scratch on
  both the driverless dev host and the GPU box, after fixing a missed `libggml-cuda.so*` copy, a
  missing `CUDA::cuda_driver`/`${GGML_CUDA_LIB}` link, and a non-transitive-RUNPATH runtime
  resolution failure (all three detailed in "Ops notes"). Real transcription confirmed on the actual
  RTX 3090 via `docker run --gpus all`, not merely a build success: the worker logged
  `ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24123 MiB): Device 0: NVIDIA GeForce RTX 3090,
  compute capability 8.6` and `[parakeet] pk::Backend using device: CUDA0`, and produced a full
  correct transcript over 20s of the real speech fixture with the real
  `realtime_eou_120m-v1-q8_0.gguf`. **Gate A re-run on CUDA** (300s, real speech, `n_threads=4`) —
  the explicitly-open Phase 1 question of whether #63 leaks host RAM or VRAM on CUDA — found an
  early ramp then a genuine plateau on *both* (RSS flat at ~4.37–4.39 GB, VRAM flat at exactly 1828
  MB, `plateau_detected: true`); numbers and the not-yet-settled caveat are in "The one constraint
  that shapes everything" above. **The nemotron multilingual path is proven end-to-end on the real
  GPU:** real `nemotron-3.5-asr-streaming-0.6b-q8_0.gguf` (fetched via `scripts/fetch_model.sh`,
  sha256 `ba2f13ec…f7a99f1`), configured with `language="en-US"` so the `stream_begin_lang` path is
  exercised (the 120M smoke test only covers plain `stream_begin`), fed the same fixture at
  nemotron's baked-in 320ms chunks, producing a real full transcript — lower quality than the 120M on
  this fixture as expected for a different model ("i need happy to kind of dot straight in" vs the
  120M's correct "i'd be happy to kind of dive straight in") — and **zero EOU events across the
  entire run**, confirming for real the "Model choice" table's claim that nemotron's special tokens
  are language tags, `eou_id_` stays -1, and turn detection must not be built on it. (The documented
  `<en-US>` tag-leak quirk was simply not exercised by this audio window — not a contradiction of
  it.) **VRAM-aware admission's read path verified against the real driver:**
  `live_stt.gpu.free_vram_mb()` inside a `--gpus all` container returned `17655`, exactly matching a
  concurrent host-side `nvidia-smi --query-gpu=memory.free` — so the exact call `servicer.py` makes
  before admission reads real, correct data in the real container runtime, which the offline
  `tests/test_gpu.py` (which only proves the no-`nvidia-smi` → `None` path) cannot show. **NOT yet
  done / NOT verified:** the **rejection** branch was deliberately never exercised — 10.100.0.50 is
  a shared box with LocalAI on the same card, and starving VRAM below the threshold to trigger it
  risks disrupting that tenant, so a scope decision, not an oversight; treat "the reject path is
  exactly right under real contention" as unverified. `vram_per_worker_mb=3000` /
  `vram_reserve_mb=2000` are **still uncalibrated guesses** — nothing measured today isolates a
  confident steady-state per-worker VRAM figure (Gate A's 1828 MB total-process growth is at least
  the right ballpark, but wasn't designed to answer that). The Gate A CUDA run was interactive
  (output to `/tmp/gate_a_cuda.log` on that box) and has **not** been promoted to a committed CSV
  artifact or a `tests/test_leak_curve.py`-style permanent pin; a 600s re-run to match the CPU
  methodology is still owed. `ggml_backend_cuda_graph_compute: CUDA graph warmup complete` was
  observed logging **once per ~160ms chunk fed** rather than capturing once and reusing — exactly
  the hazard the design plan flagged for `GGML_CUDA_GRAPHS` with variable-shaped streaming inputs.
  Recorded as an observation worth chasing (possible RTF/latency cost, **not measured**, not a proven
  problem); the documented escape hatch `LSTT_CUDA_GRAPHS=0` → `GGML_CUDA_DISABLE_GRAPHS=1` is
  **not implemented in code yet**. The VAD for synthesizing turn boundaries on the no-`<EOU>` model
  is also not built. **The full stack (not just the worker binary) was also run for real on CUDA:**
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` on 10.100.0.50 came up
  healthy with `backend: cuda` in `/api/health`; a real `scripts/e2e_grpc_smoke.py` run against it
  (gRPC client → real `CallSession` → real worker → real CUDA model, not the isolated-worker probe
  above) produced correct word-by-word deltas with timestamps and a normal `Final`
  (`worker_generations: 1`); and `docker stop` sent a real SIGTERM that drained cleanly (exit code
  0, the same `stt.server: SIGTERM received, draining` log line as the CPU path). Also: found and
  fixed a **real capacity bug** while looking at the Gate A numbers — `docker-compose.gpu.yml`'s
  `mem_limit` was `4g`, but a single worker's measured RSS (above) already plateaus around ~4.4GB
  on its own, and nothing in `live_stt/worker.py`'s `preexec_fn` bounds a worker's RSS at the
  process level (only `RLIMIT_CORE=0` is set, no `RLIMIT_AS`) — so with `max_workers` = 3 here, the
  container-wide `mem_limit` was already below what a *single* worker alone needs, let alone three.
  Raised to `20g` (~3 × 4.4GB + front-door overhead + margin); see that file's comment for the full
  arithmetic and caveats (one 300s run, not a multi-hour steady-state; concurrent-worker contention
  untested; the host's actual free-RAM headroom against its other 60+ tenants was **not**
  re-verified in this pass — an attempted `free -h` over SSH hung on a stuck forwarded-agent socket
  partway through this work and was not retried once real GPU-box access was lost).
- **Docker:** `Dockerfile` has all targets (`parakeet-build`, `worker-build`, `runtime`,
  `test-unit`, `test-integration`, `test-worker`). `test-unit`, `runtime`, and the full
  `docker-compose.yml` stack have all been built and run for real — `runtime`'s worker binary was
  run via `docker exec` against a real volume-mounted model inside the actual container (not just
  natively), confirming the `$ORIGIN` rpath resolves the ggml `.so`s correctly in that environment.
  `test-integration` is written but not yet built+run in a container — verify before trusting it
  blindly.
- **Batch ASR over HTTP (`live_stt/transcribe_http.py`) — `POST /v1/audio/transcriptions`,
  wired into `admin_http.py`'s `ThreadingHTTPServer`, real subprocess-level tests passing, not yet
  run against the real worker binary/model.** Unlike diarization (a genuinely separate, offline-only
  torch engine), this reuses the REAL production ASR path: `live_stt/session.py`'s `CallSession`
  over a real spawned `WorkerHandle` -- the exact same code `servicer.Transcribe` drives for a gRPC
  call, with one HTTP request treated as one short-lived "call" admitted through the SAME shared
  `WorkerBudget` a gRPC call uses. **Found and fixed a real concurrency bug making this safe**:
  `WorkerBudget` was documented and built as lock-free specifically because only the single-threaded
  grpc.aio event loop ever touched it; sharing it with `admin_http.py`'s per-request OS threads broke
  that argument outright (`active_calls += 1` is not atomic across real threads even under the GIL).
  Fixed by adding one `threading.Lock` around every mutating method in `live_stt/admission.py` --
  cheap (called a handful of times per call, never in a hot loop), and now this endpoint cannot spawn
  worker processes past what the box was actually sized for, independent of concurrent gRPC load.
  Long uploaded files even get `CallSession`'s normal worker-rotation safety net for free, for the
  same reason. **Request/response contract confirmed against the actual house consumer, not
  guessed**: `my-meeting-notes/app/routers/live_caption.py`'s `channel_worker_transcriptions` (the
  "reinstated stateless per-chunk POST backend" for a deployment with no realtime pipeline model) --
  multipart `file`/`model`/`stream`/`language`, and on `stream=true` (which that client always sends)
  a Server-Sent-Events response where only `{"type": "transcript.text.done", "text": ...}` is read
  and a literal `data: [DONE]` line ends it. Implemented as ONE such event carrying the full
  transcript after the whole upload is processed, not genuine incremental streaming (no
  `transcript.text.delta` mid-request) -- correct for that consumer, which never reads deltas anyway,
  but not something to build a live partial-caption UI on top of. `stream` unset/`false` returns
  OpenAI-Whisper-shaped JSON instead (`{"text": ...}`, or `response_format=verbose_json` for
  `{task, language, duration, text, words}`). Only 16kHz mono 16-bit WAV is accepted, same
  restriction `servicer.Transcribe` already enforces on the gRPC path, kept consistent rather than
  adding a resampling dependency here. Multipart parsing was extracted from `diarize_http.py` into a
  shared `live_stt/multipart.py` (nothing about it was diarization-specific) with a re-export so
  existing call sites/tests were untouched. **What's proven:** 22 tests (`tests/test_transcribe_http.py`)
  including several that go through the REAL `CallSession`/`WorkerHandle`/subprocess path against
  `tests/fakes/fake_worker_main.py` (a real subprocess speaking the real IPC protocol, same pattern
  `tests/test_session.py` already uses -- NOT the real C++ binary or a real model), and a real
  socket-level smoke test of both the `stream=true` SSE path and the plain `verbose_json` path
  against a real running `admin_http` server, producing the exact expected wire shapes. Also
  re-verified the existing diarization endpoint still routes correctly (a 503 with a clear message,
  not a hang or a 404) after the multipart refactor. **Since updated: run for real against the real
  worker binary and a real GGUF model** during the 10.100.0.50 GPU deploy (see the diarization
  entry below and the "GPU deploy" note) -- a real gRPC `Transcribe` call against the real
  `realtime_eou_120m-v1` model on CUDA produced a correct transcript over real NOTSOFAR-1 meeting
  audio (via `live_stt.client.asr_client.ASRClient`, not this HTTP endpoint directly, but exercising
  the identical `CallSession`/`WorkerHandle` code this endpoint calls into). The HTTP endpoint
  itself specifically (`/v1/audio/transcriptions`) was still not hit with a real request against the
  real worker in that pass -- only the gRPC path was at the time. **Since closed**: the actual
  redeployed 10.100.0.50 production container was hit with real `curl` multipart POSTs to this exact
  endpoint, both the plain JSON path and the `stream=true` SSE path, each producing the correct full
  transcript of a real 358s recording via the real worker/model on CUDA -- see the "Deploy status"
  paragraph in the diarization entry below. **What's still NOT proven:** never hit by an actual
  `httpx`-generated request from a real my-meeting-notes checkout (only hand-built `curl`/`urllib`
  requests so far).
- **Post-call speaker diarization (`live_stt/diarization.py`, `live_stt/diarize_http.py`,
  `tools/diarize_call.py`) — interface, HTTP endpoint, and pure mapping logic implemented,
  unit-tested, AND now run for real end-to-end against the real gated model.**
  Wraps `pyannote/speaker-diarization-community-1` (confirmed via its model card:
  `Pipeline.from_pretrained(model, token=...)`, called as `pipeline("audio.wav")` — a
  complete-clip-in-one-pass API, no incremental feed, unlike `parakeet_capi_stream_feed`). That
  mismatch is why this is batch-only and runs *after* a call ends against a recorded WAV, never
  inline in `Transcribe()` — and today that WAV has to come from somewhere out-of-band, since the
  audio-dump ring buffer this would naturally consume from doesn't exist yet (see the very next
  bullet). Deliberately **not** mapped to pyannote's native `Annotation`/RTTM shape: this house
  already has one diarization consumer, `my-meeting-notes/app/services/diarize.py` (an HTTP client
  against a LocalAI-compatible `/v1/audio/diarization` endpoint returning
  `{num_speakers, segments, speakers}` JSON with per-segment `text`), and `live_stt/models.py`
  already cites that file as prior art for the `<en-US>`-tag-stripping regex — so
  `annotation_to_house_json`/`assign_text` map pyannote's `itertracks(yield_label=True)` output into
  that exact same JSON shape, with segment `text` filled in by midpoint-overlap against the call's
  own ASR word timestamps (`asr_pb2.Word`) rather than a single model producing both, since here
  diarization (pyannote) and transcription (parakeet.cpp) are two separate engines. New `Settings`
  fields: `diarization_model`, `diarization_hf_token` (the model is gated CC-BY-4.0, kept as its own
  field so a logged model id never leaks it), `diarization_num_speakers` (originally defaulted to 2 —
  most calls through this service are one-on-one telephony, and pyannote's clustering does measurably
  better given the true count than guessing). **Since corrected — that default was itself the bug.**
  A real experiment (`tools/speaker_count_experiment.py`, results in
  `tools/speaker_count_experiment_results.json`) ran 6 configs of pyannote's actual installed
  `speaker-diarization-community-1` pipeline against a real NOTSOFAR-1 meeting with a KNOWN true
  count — `MTG_32063` (Beth/Linda/Rachel, 3 speakers, 366s, no 2-speaker meeting exists anywhere in
  this eval set) — scored against ground truth with the same frame-purity methodology as the earlier
  MTG_32089 run. `num_speakers=2` (the old hardcoded default, deliberately WRONG for this 3-speaker
  file) scored **worst of all six configs** (0.675 frame agreement, vs. 0.729 for either no hint at
  all or the exact correct hint of 3) and collapsed one real speaker's cluster purity to **0.396**
  (Beth's speech ends up attributed to the wrong name more often than not). Reading pyannote's
  installed source directly (`pyannote/audio/pipelines/clustering.py`) also confirmed
  `VBxClustering.expects_num_clusters = False` — the actual algorithm this registered model uses
  already auto-estimates the speaker count and only forces a re-cluster via `KMeans` when that
  estimate falls **outside** an explicit `[min_speakers, max_speakers]` band. Measured for real:
  `min_speakers=1, max_speakers=5` (a band bracketing the true count) produced output
  **byte-identical** to passing no hint at all — proving the band is a genuine no-op when already
  correct, not merely a theoretical claim from the docs. So: `diarization_num_speakers` now defaults
  to `None` (do not reintroduce a hardcoded exact default), and two new fields —
  `diarization_min_speakers` (default `1`) / `diarization_max_speakers` (default `6`) — carry the
  bound instead, wired through `live_stt/diarization.py::diarize_file` (num_speakers, when set, still
  takes priority and min/max are skipped entirely, matching pyannote's own contract), the
  `min_speakers`/`max_speakers` multipart fields on `POST /v1/audio/diarization`
  (`live_stt/diarize_http.py`), and `tools/diarize_call.py --min-speakers`/`--max-speakers`. An exact
  `num_speakers=` hint remains supported and is still the right choice — but only when the caller
  actually asserts a true count for that specific call, not as a blanket default. Caveat carried
  forward in the new fields' own docstrings: this is one meeting, one true-count value (3); the
  qualitative conclusion (wrong exact default is worse than no hint; a correct-or-bracketing band is
  a safe no-op) is real, but the specific band defaults `(1, 6)` are not independently tuned beyond
  it. 15 new/updated tests cover the priority ordering (`tests/test_diarization.py`,
  `tests/test_diarize_http.py`). `pyannote.audio`/`torch`/`torchaudio` live in a separate
  `requirements-diarization.txt`, deliberately excluded from `requirements.txt` — heavy dependencies
  for an opt-in offline tool, not something every deployment of the always-on `grpc.aio` server
  should pay for. `live_stt/diarize_http.py` wires `POST /v1/audio/diarization` into
  `admin_http.py`'s existing `ThreadingHTTPServer` (admin_host/admin_port, not a new server, not
  the gRPC path) — path and multipart fields (`file`/`model`/`include_text`/`response_format`)
  deliberately match `my-meeting-notes/app/services/diarize.py`'s client exactly, so that existing
  client can point at a live-stt instance with zero changes on its side. Two extension fields not
  in that client today, `words` and `num_speakers`, exist because — unlike that client's other
  backends — diarization (pyannote) and transcription (parakeet.cpp) are separate engines here, so
  per-segment `text` can only be filled if the caller supplies the call's own ASR word timestamps;
  `include_text=true` with no `words` field correctly comes back as empty-text segments, which the
  my-meeting-notes client already treats as `DiarizationError("...ignored include_text=true...")` —
  an honest, correctly-typed failure, not a silent wrong answer, but a real caller wanting text
  through this endpoint MUST also pass `words`. Since Python 3.13 removed `cgi.FieldStorage` (this
  host runs 3.14, same reason `live_stt/client/telephony.py` avoids `audioop`), multipart parsing
  goes through a documented `email`-module trick (wrap the raw body as a synthetic MIME message
  using the client's own Content-Type/boundary, let `email` split it) — verified binary-safe against
  a synthetic 1KB payload covering all 256 byte values before relying on it, and separately verified
  over a real socket (a real `urllib` multipart POST against a real running `admin_http` server
  correctly routed to the handler and returned a real 503 for the real missing-pyannote-dependency
  case, not a 404 or a crash). **What's proven:** the mapping/merge logic (13 tests,
  `tests/test_diarization.py`) against a fake `itertracks`-shaped stand-in, both label conventions
  pyannote's docs show (bare `Annotation` and the 4.x `.speaker_diarization`-wrapper output), the
  HTTP request handler (11 tests, `tests/test_diarize_http.py`, pure-function-level plus one live
  socket smoke test as above), the missing-dependency and missing-token error paths (forced via
  `sys.modules` injection rather than relying on the venv's actual state — see the real-run gotcha
  below), and that the CLI (`tools/diarize_call.py`) exits cleanly with a readable message on both.

  **Real end-to-end run, done for real (not simulated), against NOTSOFAR-1's `MTG_32089`** — the
  same meeting Gate C already used (`/data/vmfs/main01a_shared/Download/NOTSOFAR-1/eval_set/
  240629.1_eval_small_with_GT/MTG/MTG_32089`), specifically `sc_meetup_0/ch0.wav` (16kHz mono,
  358s, 5 real participants: Sarah/Donald/Ron/Beth/Rachel) plus its `gt_transcription.json`, which
  conveniently has real per-speaker ground truth AND real word-level timestamps (1151 words) —
  used the latter as a stand-in for "the call's own ASR word list" without needing the C++ worker
  at all. `requirements-diarization.txt` installed clean into this repo's venv (torch 2.13.0,
  pyannote.audio 4.0.7, confirming the `>=4.0.0` floor pin). HF token (real, accepted-terms account)
  stored at `~/.secrets/huggingface.env` as `LSTT_DIARIZATION_HF_TOKEN`, sourced from `.zshenv` —
  same convention as `BAMBOO_TOKEN` in the global CLAUDE.md, deliberately not typed inline anywhere.
  **CLI path** (`tools/diarize_call.py --num-speakers 5`): real pyannote inference, **5/5 speakers
  correctly counted**, 130 segments, 312.8s wall time on this 6-core CPU-only dev host (~447% CPU,
  no GPU) — call this roughly real-time-ish for a ~6-minute meeting on CPU, not fast, and NOT yet
  measured on the actual deployment box or on CUDA. Word-timestamp merge worked on real (not
  synthetic) data: 96/130 segments got real, correctly-attached transcript text (spot-checked, e.g.
  segment 3 = `"so uh rachel you're here just to like yes give us a rundown..."`, the real words in
  the real time range). **Scored against real ground truth** with a simple 100ms-frame purity metric
  (not a real DER — no optimal/Hungarian assignment, no miss/false-alarm penalty, sanity-check-grade
  only): the 5 predicted clusters mapped **cleanly and bijectively** to the 5 real speakers (no
  cluster split across two identities as its dominant match), per-cluster purity 0.73–0.86, overall
  frame-level agreement **0.813** across 288.1s of overlapping scored audio. **HTTP path**: the exact
  same audio+words sent as a real multipart POST over a real TCP socket to a real running
  `admin_http` server → **HTTP 200 in 361.7s**, and the returned JSON scored **identically** (0.813)
  to the CLI path, confirming the two code paths agree byte-for-byte in outcome, not just in shape.
  **A real gotcha this run surfaced**: two unit tests were silently depending on ambient
  environment state that stopped being true the moment the real dependency and real token existed
  on this dev host — `test_missing_dependency_raises_diarization_error` assumed pyannote.audio was
  actually absent (broke the instant it got installed for this run) and `test_diarization_defaults`
  assumed `LSTT_DIARIZATION_HF_TOKEN` was unset (broke the instant `.zshenv` started sourcing the
  real one) — `Settings(_env_file=None)` only skips the `.env` FILE, not real process env vars, and
  `tests/conftest.py`'s offline-safety fixture cleared `LSTT_MODEL_PATH`/`LSTT_ALLOW_PII` but not
  this new var. Fixed by forcing the import failure via `sys.modules["pyannote.audio"] = None`
  (asserts the code path, not the venv's current state) and adding
  `monkeypatch.delenv("LSTT_DIARIZATION_HF_TOKEN")` to that same fixture — the same class of bug the
  fixture already exists to prevent, just not yet extended to this field.

  **GPU support added (`Settings.diarization_device`, "cpu" default | "cuda") and measured for real
  on 10.100.0.50's actual RTX 3090** — not simulated, not assumed from the ASR worker's separate CUDA
  numbers. `load_pipeline` moves the pipeline with `pipeline.to(torch.device("cuda"))`, checking
  `torch.cuda.is_available()` first and failing loudly (a clear `DiarizationError`) rather than
  letting a missing GPU surface as a confusing failure deep inside pyannote's first forward pass —
  this is independent of the ASR worker's own `backend` setting, since diarization runs as ordinary
  Python/torch in this process, never in the C++ worker. **Deployment note: this was validated via a
  standalone venv on 10.100.0.50 (`~/live-stt-diarize-test`, python3.14-venv installed via
  `sudo apt-get install`, `requirements-diarization.txt` installed fresh against the real CUDA
  driver there), NOT by rebuilding/redeploying the actual running `live-stt` Docker container** (the
  one `docker ps` shows healthy on that box today, per the user's note that its ASR backend is
  currently configured `cpu` there too) — wiring GPU diarization into that container's own image and
  compose file is separate, not-yet-done work. Same exact file (`MTG_32089/sc_meetup_0/ch0.wav`,
  already reachable at the same path on 10.100.0.50 too — `main01a_shared` is genuinely shared
  network storage, no copy needed), same `words.json`, same `--num-speakers 5`, run four ways for a
  clean CPU-vs-GPU, CLI-vs-HTTP comparison:

  | path | device | wall time | speakers | segments | accuracy vs. ground truth |
  |---|---|---|---|---|---|
  | CLI | cpu (dev host, 6-core) | 312.8s | 5/5 | 130 | 0.813 frame agreement |
  | CLI | cuda (10.100.0.50, RTX 3090) | **16.5s** | 5/5 | 130 | 0.813 (identical) |
  | HTTP | cpu (dev host) | 361.7s | 5/5 | 130 | 0.813 (identical) |
  | HTTP | cuda (10.100.0.50) | **10.8s** | 5/5 | 130 | 0.813 (identical) |

  **~19x wall-time speedup, zero accuracy difference** — segment boundaries matched CPU output to
  the float (max start-time delta across all 130 segments: 0.0s), despite pyannote disabling TF32 on
  CUDA specifically for reproducibility (a real warning seen in this run's own logs:
  `TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues`). This turns
  post-call diarization from "roughly as long as the call itself" (CPU, the caveat given right after
  the CPU timing question was first asked) into a genuinely fast background step on hardware this
  house already owns. **What's still NOT proven:** this was a 5-party conference meeting run with
  `--num-speakers 5` explicitly, not the 2-party telephony call `diarization_num_speakers`'s default
  (2) actually targets — the default path itself is untested against real audio; the HTTP request
  was a hand-built multipart POST matching my-meeting-notes' client's contract, not that client's
  actual code hitting this endpoint; CPU-only timing on the 6-core dev host is not a production
  capacity number, the same caveat Gate B/C's numbers already carry elsewhere in this file; GPU VRAM
  used by pyannote itself was not isolated/measured (10.100.0.50 is the same shared card
  `vram_per_worker_mb`/`vram_reserve_mb` already budget around for the ASR worker — a real GPU
  diarization deployment there would need its own VRAM accounting, not proven here). (The venv vs.
  container gap noted here originally is now closed -- see the next paragraph.)

  **Real container image built and verified for real GPU deployment (still not yet cut over in
  production as of this writing).** New Dockerfile stage `runtime-cuda-diarize` (extends
  `runtime-cuda`, adds `requirements-diarization.txt`; `build.sh --cuda --diarize` builds it, tags
  get a `-cuda-diarize` suffix). **Found a real, container-specific bug the venv testing above could
  not have caught**: pyannote.audio 4.x decodes audio via `torchcodec`, which `dlopen`s
  `libavutil`/`libavcodec`/etc at import time rather than linking them at build time --
  `runtime-cuda`'s minimal base (`python3.12`/`python3-pip`/`libgomp1` only) has none of that, so the
  first real container run failed with `OSError: libavutil.so.61: cannot open shared object file`
  (repeated for every ffmpeg ABI version 4-9 torchcodec tried). Every dev machine this was built on
  before this container already had `ffmpeg` installed for unrelated reasons, which is exactly how
  this stayed invisible until a real container build+run. Fixed by adding `ffmpeg` via `apt-get` in
  the new stage. Also: `HF_HOME` is set to `/app/data/hf-cache`, inside the already-mounted,
  already-writable `live-stt-data:/app/data` volume -- the gated model then survives container
  restarts using infrastructure that already exists, rather than a container-local cache that
  re-downloads it (burning the token's rate limit) every time. **After the ffmpeg fix, verified for
  real on 10.100.0.50, in the actual container (not a venv), with `--gpus all`**: real gRPC ASR
  transcript over CUDA against real NOTSOFAR-1 meeting audio (via `ASRClient`, exercising the same
  `CallSession`/`WorkerHandle` code `servicer.Transcribe` and the `/v1/audio/transcriptions` HTTP
  endpoint both call into); real pyannote diarization over CUDA via both the direct `diarize_file()`
  call and a real `curl` multipart POST to `/v1/audio/diarization` on the container's actual admin
  port -- 5/5 speakers, 130 segments, matching every prior CPU and venv-GPU run exactly; and the
  pyannote model persisting for real to the host path
  (`/data/docker-infra-data-vol-ssd/live-stt-data/hf-cache/hub/models--pyannote--speaker-diarization-community-1/`).
  **Deploy status: DONE, cut over for real, verified against the actual redeployed production
  container** (this was blocked mid-edit by a stuck forwarded-SSH-agent socket to that host --
  the same class of issue Phase 5's `free -h` attempt hit earlier in this file -- and finished once
  that was reconnected; not anything wrong with the image or the plan). `ai.yml`'s live-stt service:
  `image:` swapped to `registry.shifamily.com/homestack/live-stt:latest-cuda-diarize` (built and
  sitting locally on 10.100.0.50, never pushed -- no registry credentials configured on that host,
  and none needed since compose finds the matching local tag without pulling), `environment:` gained
  `LSTT_MAX_CONCURRENT_CALLS=2` alongside the existing `LSTT_RESERVE_SLOTS=1` (`ai.env`'s plain `=20`
  is the CPU-path number). `ai.env` already had the GPU toggle uncommented and the diarization
  token added from the earlier pass. Deployed with `./homestack ai up -d live-stt` (recreated only
  that one service, no other stack service touched). **Verified against the real, running,
  redeployed production container** (not a side-channel container, not a venv): `/api/health` and
  `/api/stats` on the real exposed admin port (4031) report `backend: cuda`,
  `max_concurrent_calls: 2`, `max_workers: 3`; a real gRPC `Transcribe` call over the real exposed
  gRPC port (4030) against the full NOTSOFAR-1 meeting audio produced the same correct transcript as
  every earlier test; a real `curl` multipart POST to `/v1/audio/diarization` on port 4031 returned
  5/5 speakers and 130 segments in ~13.9s, matching every prior run exactly; and real `curl` POSTs to
  `/v1/audio/transcriptions` — both the plain JSON path and the `stream=true` SSE path — each
  produced the correct full transcript of the entire 358s recording, closing the one gap this
  entry's own "Batch ASR over HTTP" counterpart still listed as unverified (that endpoint against a
  real worker/model). Container has been stable and healthy for several minutes post-deploy with no
  restarts.
- **Diarization VRAM admission + GPU/diarization visibility in the admin dashboard
  (`live_stt/gpu.py`, `live_stt/diarize_sessions.py`, `live_stt/diarize_http.py`,
  `live_stt/admin_http.py`) — unit-tested for real, not yet redeployed to 10.100.0.50.**
  Prompted by a real measurement: `nvidia-smi --query-compute-apps` against the actual redeployed
  production container (see the "Deploy status" paragraph above) found **12,312 MiB used, isolated
  to the live-stt process's own PID**, after a single diarization request over the ~6-minute
  NOTSOFAR-1 meeting — baseline was 41 MiB. A second back-to-back request on the same file used
  essentially the same VRAM (12,320 MiB) and was *faster* (8.8s vs 13.7s), not slower or larger:
  the signature of PyTorch's CUDA caching allocator sizing itself once to the batched
  sliding-window peak for that file and reusing the pool, not a leak. The pyannote model itself is
  tens of MB (confirmed earlier: the pre-warmed HF cache is ~32MB) — the 12GB is allocator/activation
  overhead, not model weights. Caveat carried into the new setting's own docstring: measured on ONE
  358s file: VRAM plausibly scales with audio duration (bigger batched windows for longer
  recordings), untested across a range of durations.

  `live_stt.gpu` gained `total_vram_mb()`/`used_vram_mb()`/`utilization_pct()`/`snapshot()` alongside
  the existing `free_vram_mb()` (all `_query_gpu()`-backed, same None-on-unavailable contract, and
  `snapshot()` is guaranteed all-None together, never a partial mix, since a nvidia-smi failure mode
  isn't field-specific). `Settings.diarization_vram_mb` (default 13000, ~700MB margin above the one
  real measurement) is checked against `gpu.free_vram_mb()` in `diarize_http.handle_diarize_request`
  before running the pipeline, the same admission pattern `servicer.py` already uses for ASR — and
  for the same reason: a CUDA allocation failure inside pyannote/torch is not guaranteed to be a
  catchable Python exception any more than parakeet.cpp's is. Fails OPEN (admits) when
  `free_vram_mb()` returns `None` (nvidia-smi unavailable = "cannot check", never "zero free" — same
  contract the ASR path already relies on), and does nothing at all on the `"cpu"` default device
  (no VRAM to check). A new `DiarizationSessionTracker` (`live_stt/diarize_sessions.py`) — a
  thread-safe aggregate counter, explicitly NOT a session registry (same category as
  `WorkerBudget`'s own `active_calls`/`active_workers`, consistent with `live_stt/state.py`'s
  "Deliberately NOT a session registry" invariant) — now lives on `ServerState` and tracks
  active/completed/failed/VRAM-rejected diarization requests, exposed at `/api/stats` under a new
  `"diarization"` key and via two new Prometheus metrics
  (`live_stt_diarization_sessions_active`, `live_stt_diarization_requests_total{outcome}`).
  `/api/stats` also gained a `"gpu"` key (`gpu.snapshot()`) and the admin HTML dashboard got four new
  cards (GPU VRAM free/total, GPU utilization, active diarization sessions, diarization
  completed/failed/rejected-VRAM totals) plus a VRAM usage progress bar, all null-safe when no GPU
  is present (renders "N/A", hides the VRAM bar row entirely rather than showing "NaN%"/"undefined").
  **What's proven:** 17 new tests (`tests/test_gpu.py`, `tests/test_diarize_sessions.py`, new cases
  in `tests/test_diarize_http.py`, and a new `/api/stats` JSON-shape test plus a dashboard-markup
  test in `tests/test_metrics.py`) — including the VRAM-insufficient-rejects-before-touching-the-
  pipeline path, the fail-open-on-None path, and thread-safety of the tracker under concurrent
  start/finish. **What's NOT proven:** the dashboard's actual rendering (vs. just the HTML/JSON
  shape tests above) has not been eyeballed in a real browser against a real backend.

  **Duration-scaling measured for real, and a real leak-shaped (not-actually-a-leak) finding fixed.**
  Concatenated real meeting audio into synthetic 10/20/40-minute files (no natural NOTSOFAR-1
  recording runs past ~8 minutes) and measured peak VRAM against the real RTX 3090, restarting the
  container fresh before each duration so one file's plateau couldn't contaminate the next:

  | duration | peak VRAM (this process's own PID) |
  |---|---|
  | 6 min | 10,796 MiB |
  | 10 min | 11,424 MiB |
  | 20 min | 10,320 MiB |
  | 40 min | **2,328 MiB** (reproduced twice, with continuous 0.5s polling both times) |

  **VRAM does not scale up with call duration** in this tested range — 6 to 20 minutes (a 3.3x
  spread) stayed within ~10% of each other, consistent with pyannote batching sliding windows at a
  bounded size internally rather than batching "the whole file" (which would show duration-scaling).
  The 40-minute result being *lower*, not higher, is real and reproducible but its mechanism is
  NOT understood (best guess: length-aware batching inside pyannote itself, unconfirmed against its
  source). Practically: the original fear motivating this whole investigation (a long real phone
  call blowing past 24GB) is not supported by this data, though it was only tested to 40 minutes,
  not this service's full 150-minute `max_call_sec` ceiling.

  Separately, and prompted directly by the question "why doesn't it release the VRAM after the job
  finishes": confirmed the pre-existing deployed image had NO cleanup between requests at all --
  `diarize_file` now calls `gc.collect()` + `torch.cuda.empty_cache()` in a `finally` after every
  CUDA request (`live_stt/diarization.py::_release_cuda_memory`), trading some re-warm cost on the
  next request for not permanently squatting on a shared card between requests. **Verified for real
  on 10.100.0.50** (rebuilt the image, real GPU, real requests): peak VRAM during a request dropped
  back from ~10.8GB to **370-398 MiB** (bare CUDA context) within ~1s of the request completing, and
  stayed there rather than creeping back up on a second call -- and that second call was still
  *faster* than the first (9.3s vs 14.0s), not slower, meaning cuDNN's algorithm-selection cache
  (unaffected by `empty_cache()`, which only touches memory) still gives the repeat-call speedup
  with none of the idle-VRAM cost. One genuinely useful side effect of testing this: the *old*,
  not-yet-redeployed production container's own 12GB resting footprint (still running the
  pre-fix image) blocked the new VRAM gate from admitting the test container at the default 13000MB
  threshold -- a live demonstration, on the real box, of exactly the problem this fix solves.
  **Deploy status: same as the VRAM-gate/dashboard work above -- built and verified in a throwaway
  container, NOT yet redeployed to production.**
- **Diarization model registry (`live_stt/diarization_models.py`) — closes a real validation gap,
  unit-tested, not yet redeployed.** Before this, `Settings.diarization_model` was a bare string
  passed straight to `pyannote.audio.Pipeline.from_pretrained` with zero validation — unlike ASR's
  `live_stt/models.py::resolve()`, which rejects an unknown key with a clear error before ever
  touching the engine. Mirrors that pattern exactly: `DIARIZATION_MODELS: dict[str,
  DiarizationModelSpec]`, `DEFAULT_DIARIZATION_MODEL_KEY`, `resolve()` raising `KeyError` on an
  unknown key. `load_pipeline` now resolves and validates FIRST, before even attempting the
  pyannote.audio import — same "cheap check before the heavy one" ordering ASR already uses — so a
  typo'd model name fails fast regardless of whether the dependency is installed. `diarize_http.py`
  needed **zero changes** for this to produce the right HTTP status: an "unknown diarization model"
  `DiarizationError` doesn't match that handler's existing `"not installed"`/`"HF_TOKEN"`/`"Failed
  to load"` substrings, so it already fell through to the existing `else 400` branch correctly.
  `DiarizationModelSpec` carries real metadata beyond just the HF repo id: `gated` (only requires
  `LSTT_DIARIZATION_HF_TOKEN` if the specific model actually needs one — the one registered model,
  `pyannote/speaker-diarization-community-1`, does), `supports_num_speakers_hint` (only passes
  `num_speakers=` to the pipeline if the model actually accepts it), and `measured_peak_vram_mb`
  (11424 — the highest of the real 6/10/20-minute measurements above; the 40-minute result was
  reproducibly *lower*, not a worse case, so it's deliberately excluded from "peak"). That VRAM
  figure is informative metadata only, not yet wired into a per-model admission threshold —
  `Settings.diarization_vram_mb` stays one global, operator-tunable value regardless of which model
  is selected, since a per-model lookup isn't worth the complexity with only one registry entry.
  **What's proven:** 11 new tests (`tests/test_diarization_models.py` plus new cases in
  `tests/test_diarization.py`) covering unknown-key rejection (including that it happens before
  pyannote.audio is ever imported), a non-gated fake model skipping the token check, and a
  no-hint fake model never receiving `num_speakers=` even when `diarization_num_speakers` is
  configured. **What's NOT proven:** not yet redeployed to 10.100.0.50 — same status as the
  VRAM-gate/dashboard/release-fix work above, all still sitting in throwaway test containers.
- **Not yet built:** the AudioRing/backpressure design and its drift watchdog (`queue_max_sec`,
  `ring_history_sec`, `warn_behind_sec`, `abort_behind_sec` exist as `Settings` fields, referenced
  in a docstring, but nothing reads them yet — `feed_audio()` just buffers to one model chunk and
  calls the worker; there is no bounded ring, no drop-oldest policy, and no `behind_sec`
  computation). `LSTT_AUDIO_DUMP` is validated at startup but has nothing to act on (no ring buffer
  to dump from). Sending a live `Warning{SERVER_DRAINING}` event into an *already-open* stream when
  a drain starts is also not implemented — draining currently only blocks *new* admission; an
  active call gets no in-band notice that a deadline is coming, it just has up to
  `drain_timeout_sec` to finish naturally before `grpc.aio`'s own stop-grace forcibly ends it.
  Prometheus scrape-job/Grafana dashboard JSON, long-call and concurrency tests: not written yet
  either. (The CUDA build is no longer on this list — see the Phase 5 entry above.)

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
- **The GPU box is 10.100.0.50, and its hardware was WRONG in earlier drafts of this file — verified
  for real via `nvidia-smi` over SSH, not inferred.** It is one **NVIDIA GeForce RTX 3090 (24GB,
  Ampere, sm_86)**, driver 595.84 / CUDA 13.2, `nvidia-container-toolkit` present and
  `docker info` reports the `nvidia` runtime registered. Earlier speculation about an RTX 5090
  (Blackwell, sm_120) came from an unrelated repo's Best Buy stock-tracker (`check-gpu-cli`) and
  was never actually checked against the real box — a reminder that circumstantial evidence in
  someone's unrelated tooling is not verification. **`CMAKE_CUDA_ARCHITECTURES` must be `86`, not
  `120` or `90;120`** — those targets would silently fail to run (or need PTX JIT with startup
  latency, if they run at all) on this actual card. That box is also a **shared home server**: 60+
  containers, 32 cores, 121GB RAM (often only ~15GB "available"), and the GPU already has another
  tenant using ~6GB VRAM plus LocalAI on the same card — `LSTT_VRAM_PER_WORKER_MB`/
  `LSTT_VRAM_RESERVE_MB` (`live_stt/gpu.py`, `live_stt/config.py`) exist because of this, not as a
  generic best practice.
- CUDA Dockerfile targets (`parakeet-build-cuda`, `worker-build-cuda`, `runtime-cuda`) exist and
  build cleanly from scratch, verified both on the driverless dev host (6 cores, ~469s for the
  parakeet CUDA stage alone) and on 10.100.0.50 itself (32 cores, faster) — compiling needs only the
  CUDA toolkit, not a real device/driver. **Found and fixed while building them: a Docker BuildKit cache
  mount collision** — `parakeet-build` and `parakeet-build-cuda` both used
  `--mount=type=cache,target=/build` with no explicit `id=`, which is scoped by target path alone
  and can collide with an UNRELATED Dockerfile elsewhere on the same host also using `/build`. That
  leaked a stale `CMakeCache.txt` in from a plain `ubuntu:24.04` context (gcc-14 default) into this
  image (gcc-13 only, per the `nvidia/cuda:12.8.1-devel-ubuntu24.04` base), producing a baffling
  `No rule to make target '/usr/lib/gcc/x86_64-linux-gnu/14/libgomp.so'` link failure that had
  nothing to do with CUDA at all. Fixed by giving each stage's cache mount a unique `id=`. **Also
  found: `scripts/build_worker.sh`'s `--cuda` flag had a latent, never-executed shell bug** —
  `-DCMAKE_CUDA_ARCHITECTURES=90;120` unquoted, where the shell would have parsed the `;` as a
  command separator and broken the script the first time anyone actually ran it.
- **Three more CUDA build bugs, each found only by actually building the chain end to end** (all
  three fixed; none would have been caught by reading the code, and each one hid behind the previous
  one):
  - **`libggml-cuda.so*` was never copied out of `parakeet-build-cuda` at all.** The `cp -a
    .../ggml/src/libggml*.so* /out/lib/` glob does not match it, because unlike `libggml-cpu.so`
    (directly in `third_party/ggml/src/`) the CUDA backend builds **one directory deeper**, at
    `third_party/ggml/src/ggml-cuda/libggml-cuda.so*`. The failure mode is the nasty part: that
    stage still *succeeded*, with a plausible-looking 3-lib `/out/lib`, and broke the **next** stage
    instead — `worker-build-cuda` failing at link time with `undefined reference to
    'ggml_backend_cuda_reg'`. Fixed with a second explicit `cp -a .../ggml-cuda/libggml-cuda.so*
    /out/lib/`.
  - **`worker/CMakeLists.txt`'s CUDA comment was wrong, and the fix uncovered a second link
    error.** The comment claimed ggml's backend registry `dlopen`s `libggml-cuda.so` by soname at
    runtime and so "is never a direct link-time symbol dependency" — disproven by testing:
    `ggml-backend-reg.cpp` is compiled into `libggml.so` with a **direct** (non-weak, non-`dlsym`)
    reference to `ggml_backend_cuda_reg()` whenever the ggml build had CUDA enabled. A shared
    library tolerates that unresolved at *its* link; our executable does not — so omitting
    `target_link_libraries(live_stt_worker PRIVATE ${GGML_CUDA_LIB})` is a hard link error, not a
    co-location nicety. Linking it then surfaced `libggml-cuda.so: undefined reference to
    'cuMemCreate'` (also `cuDeviceGet`, `cuMemAddressReserve`, …) — CUDA **Driver** API symbols
    living in `libcuda.so.1`, shipped by the NVIDIA *driver*, which does not exist on a driverless
    build host. Fixed with `find_package(CUDAToolkit REQUIRED)` +
    `target_link_libraries(live_stt_worker PRIVATE CUDA::cuda_driver)`, which resolves to the
    toolkit's link-time **stub** (`/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so`,
    located with `find / -iname 'libcuda.so*'` inside the devel image) when no driver is present,
    and to the real `libcuda.so.1` at container run time on the GPU box.
  - **`$ORIGIN` rpath is not enough on CUDA: RUNPATH is not transitively inherited.** After the
    above two fixes the `runtime-cuda` binary still failed at **runtime** — `ldd` showing
    `libggml-cuda.so.0 => not found` with the file sitting right beside the binary in
    `/app/worker/`. Cause: `libggml.so.0` (built by ggml's own separate CMake during
    `parakeet-build-cuda`) has a **hardcoded RUNPATH baked in at its own build time** pointing at
    the build cache-mount path (`/build/parakeet/third_party/ggml/src:.../ggml-cuda`) — correct
    there, meaningless in the final image. And RUNPATH (the modern default, unlike old-style RPATH)
    is **not** searched for a dependency's own dependencies: the worker binary's `$ORIGIN` RUNPATH
    resolves only *its* direct `NEEDED` entries, not the ones `libggml.so.0` declares. This class
    of bug is **CUDA-specific** — the CPU-only `libggml.so` has no CUDA backend to need, so it never
    arises there, which is exactly why the `runtime` stage's `$ORIGIN` verification did not predict
    it. Fixed with `ENV LD_LIBRARY_PATH=/app/worker` on the `runtime-cuda` stage —
    `LD_LIBRARY_PATH`, unlike RUNPATH, *is* searched transitively for all dependencies regardless of
    RUNPATH scoping. Verified by `ldd` before and after: afterwards everything resolves except
    `libcuda.so.1`, which correctly stays "not found" on the driverless dev host and resolves under
    `docker run --gpus all` on 10.100.0.50 (confirmed there).

## Implementation phases (see the original plan for full detail)

0. **Skeleton + first light — DONE.** Worker builds and runs against a real model; gRPC/health/
   reflection/GetServerInfo work; `test-unit` Docker target builds.
1. **Measure before designing the pool — DONE.** `tools/leak_curve.py` (CPU; CUDA re-run done in
   Phase 5, 300s, plateau found — not yet pinned by a test), `tools/thread_sweep.py`, the
   telephony-band WER penalty test — all three run for real, with recorded numbers and permanent
   regression tests. See "Phase 1 measurements" above.
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
5. **GPU and multilingual — mostly done on real hardware, but three loose ends before trusting it
   in production.** The CUDA build chain (`runtime-cuda`, `docker-compose.gpu.yml`) builds clean
   after three real bugs found by building it (see "Ops notes"); real transcription, Gate A
   (plateau, not a linear leak), the nemotron `stream_begin_lang` path (zero EOU events, confirmed),
   and `gpu.free_vram_mb()` against the real driver (`17655`, matching `nvidia-smi`) are all
   verified for real on the RTX 3090 — details in the Phase 5 entry above. Loose ends: (a) VRAM
   admission's **reject** branch is untested for real (deliberately — shared box, LocalAI on the
   same card) and `vram_per_worker_mb`/`vram_reserve_mb` remain uncalibrated guesses; (b) the CUDA
   Gate A run is 300s, interactive, unpromoted to a committed artifact or a regression pin — a 600s
   re-run is owed before trusting long CUDA calls without rotation; (c) CUDA graph capture appears
   to re-warm per ~160ms chunk (cost unmeasured) and the `LSTT_CUDA_GRAPHS=0` escape hatch is not
   implemented. Not started at all: a VAD to synthesize turn boundaries for the no-`<EOU>` model.
