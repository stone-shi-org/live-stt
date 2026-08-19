# AGENTS.md

See [CLAUDE.md](CLAUDE.md) for architecture, the upstream leak that shapes the whole design, and
the non-obvious failure modes worth knowing before changing anything.

Quick reference:

- Build the C++ worker: `./scripts/build_worker.sh` → `worker/build/live_stt_worker`
- Fetch a model: `./scripts/fetch_model.sh` → `./models/*.gguf`
- Run the server natively: `venv/bin/python run.py`
- Test: `./test.sh` (unit) · `./test.sh --docker` (canonical — avoids the Python 3.14 grpcio
  wheel gap) · `./test.sh --integration` (real worker + model, needs the two steps above first)
- Build the image: `./build.sh [--cuda]`
