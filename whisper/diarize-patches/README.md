# diarize-patches

Local modifications to `notch-up/diarize`, applied at Docker build time
by overwriting files in a pinned upstream checkout.

## Contents

- `embeddings.py` — parallelised version of `src/diarize/embeddings.py`.
  The upstream window loop runs single-threaded and pins one CPU core
  even on multi-hour audio. This version runs the same loop across a
  `ThreadPoolExecutor`, with one WeSpeaker session per worker thread.
  Output ordering is preserved, so downstream clustering sees an
  identical `(embeddings, subsegments)` pair.

  Override worker count with the `DIARIZE_EMBEDDING_WORKERS` environment
  variable (integer ≥ 1). Default is `os.cpu_count()`. Set to `1` to
  restore the original serial behaviour for A/B benchmarking.

## How it's applied

Each `Dockerfile.{cpu,nvidia,rocm}` does, inside the pip-install step:

1. `git clone --depth 1 --branch <DIARIZE_REF> https://github.com/notch-up/diarize.git /tmp/diarize`
2. `cp /tmp/diarize-patches/embeddings.py /tmp/diarize/src/diarize/embeddings.py`
3. `pip3 install --no-cache-dir /tmp/diarize`

`DIARIZE_REF` is a build arg, defaulting to `v0.1.2`. Bump it when you
want to move to a newer upstream release — the patch must still apply
(the target file `src/diarize/embeddings.py` must still exist and have
the same imports and constants). If the file has moved or been renamed
upstream, this whole approach needs revisiting.

## Updating

When upstream releases a new version:

1. Bump `DIARIZE_REF` in the three Dockerfiles (or pass `--build-arg`).
2. Clone the new tag locally, compare its `src/diarize/embeddings.py`
   against our patched version, and reconcile any changes.
3. Commit both the Dockerfile change and any updates to `embeddings.py`.