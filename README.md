# live-stt

Streaming ASR service wrapping [parakeet.cpp](https://github.com/mudler/parakeet.cpp)'s
streaming C API, exposed over bidirectional gRPC:

```
one phone call = one gRPC Transcribe stream = one logical ASR session
```

See [CLAUDE.md](CLAUDE.md) for architecture, the upstream leak that shapes the whole design
(issue [mudler/parakeet.cpp#63](https://github.com/mudler/parakeet.cpp/issues/63)), and the
non-obvious failure modes worth knowing before changing anything.

Quick reference:

- Build: `./build.sh`
- Test: `./test.sh` (unit, native/offline) · `./test.sh --docker` (canonical path — avoids the
  Python 3.14 `grpcio` wheel problem) · `./test.sh --integration` (needs `./models/*.gguf`)
- Fetch models: `./scripts/fetch_model.sh`
- Run: `docker compose up -d --build`
