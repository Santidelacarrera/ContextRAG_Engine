# Changelog

## Unreleased

- Windows portability: `load_bench` no longer imports POSIX-only `resource` at module load; secret-file mode check applies
  on POSIX only; CLI `keygen --add-to` reads/writes UTF-8; the test suite runs on Windows.
- Repository: `.gitattributes` forces LF; the stale tracked `build/` output was removed.
- Deployment: image sets `PYTHONUNBUFFERED`, healthcheck with timeout/start period, `--proxy-headers`, graceful
  shutdown; compose adds `restart`, read-only root filesystem, `cap_drop: ALL`, `no-new-privileges`, memory/pids limits,
  a read-only mount for the keys file and a volume for the conversational-memory database; `.dockerignore`.
- `qdrant-client` is capped to versions compatible with the pinned Qdrant server image (`<1.14`).
- CI: least-privilege token, concurrency, timeouts, `bandit`, and a job that builds the image and probes `/healthz`.
- Added `SECURITY.md` and Dependabot configuration.
