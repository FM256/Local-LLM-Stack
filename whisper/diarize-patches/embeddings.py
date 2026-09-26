"""Speaker embedding extraction using WeSpeaker ResNet34-LM (ONNX).

Extracts 256-dimensional speaker embeddings from audio segments detected
by VAD.  Long segments are split with a sliding window so that each
window produces its own embedding, improving clustering granularity.

The window loop runs across a thread pool (one WeSpeaker session per
worker thread) so that embedding extraction is not limited to a single
CPU core.  Set ``DIARIZE_EMBEDDING_WORKERS=1`` to recover the original
serial behaviour.

This file is a local patch of upstream ``diarize`` (notch-up/diarize).
It is copied over ``src/diarize/embeddings.py`` during the Docker build;
see whisper/diarize-patches/README.md.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from .utils import SpeechSegment, SubSegment

logger = logging.getLogger(__name__)

__all__ = ["extract_embeddings"]

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum segment duration for embedding extraction (seconds).
#: Segments shorter than this are skipped during embedding extraction
#: and later assigned the nearest speaker label.
MIN_SEGMENT_DURATION: float = 0.4

#: Sliding window length for splitting long segments (seconds).
EMBEDDING_WINDOW: float = 1.2

#: Sliding window step size (seconds).  Overlap = WINDOW − STEP.
EMBEDDING_STEP: float = 0.6

#: Emit a progress log record every N processed windows.  Consumed by
#: the SSE bridge in openai_proxy.py to surface live diarization
#: progress to the client.
PROGRESS_EVERY: int = 200


def _resolve_workers() -> int:
    """Number of worker threads for embedding extraction.

    Reads ``DIARIZE_EMBEDDING_WORKERS`` if set (integer). Otherwise uses
    ``os.cpu_count()``.  A value of 1 forces the original serial path,
    which is useful for A/B benchmarks.
    """
    raw = os.environ.get("DIARIZE_EMBEDDING_WORKERS")
    if raw is not None:
        try:
            n = int(raw)
            if n >= 1:
                return n
        except ValueError:
            logger.warning(
                "DIARIZE_EMBEDDING_WORKERS=%r is not an integer; using cpu_count()",
                raw,
            )
    return max(1, os.cpu_count() or 1)


def _build_windows(
    speech_segments: list[SpeechSegment],
) -> list[tuple[float, float, int]]:
    """Flatten speech segments into a list of (start, end, parent_idx) windows.

    ``parent_idx`` is the index of the originating segment in
    ``speech_segments`` and is used to populate ``SubSegment.parent_idx``.
    """
    windows: list[tuple[float, float, int]] = []
    for idx, seg in enumerate(speech_segments):
        seg_duration = seg.duration
        if seg_duration < MIN_SEGMENT_DURATION:
            continue
        if seg_duration <= EMBEDDING_WINDOW * 1.5:
            windows.append((seg.start, seg.end, idx))
        else:
            win_start = seg.start
            while win_start + MIN_SEGMENT_DURATION < seg.end:
                win_end = min(win_start + EMBEDDING_WINDOW, seg.end)
                windows.append((win_start, win_end, idx))
                win_start += EMBEDDING_STEP
    return windows


def extract_embeddings(
    audio_path: str | Path,
    speech_segments: list[SpeechSegment],
) -> tuple[np.ndarray, list[SubSegment]]:
    """Extract 256-dim speaker embeddings using WeSpeaker ResNet34-LM (ONNX).

    Long segments are split using a sliding window for more accurate
    clustering.  Each window produces its own embedding.  The window
    loop is parallelised across a thread pool; results preserve the
    original window ordering so downstream clustering sees the same
    correspondence between embeddings and subsegments as before.

    Args:
        audio_path: Path to the audio file (wav, mp3, flac, etc.).
        speech_segments: Speech segments detected by VAD.

    Returns:
        A ``(embeddings, subsegments)`` tuple where:

        -   **embeddings** --- ``np.ndarray`` of shape ``(N, 256)`` with
            raw speaker embeddings (not yet L2-normalised; normalisation
            is applied later during clustering).
        -   **subsegments** --- list of :class:`SubSegment` objects that
            record the time window and parent segment index for each
            embedding row.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.

    Example::

        from diarize.vad import run_vad
        from diarize.embeddings import extract_embeddings

        segments = run_vad("meeting.wav")
        embeddings, subs = extract_embeddings("meeting.wav", segments)
        print(embeddings.shape)  # (N, 256)
    """
    import wespeakerruntime as wespeaker_rt

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    workers = _resolve_workers()
    logger.info(
        "Extracting speaker embeddings (WeSpeaker ResNet34-LM, 256-dim) "
        "with %d worker(s)...",
        workers,
    )

    # Load full audio for segment slicing.  Soundfile decodes the whole
    # file into memory; for multi-hour inputs this is a few hundred MB
    # of float32.  Acceptable for the target use case; streaming decode
    # would be a much larger change.
    audio_data, sr = sf.read(str(audio_path))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # stereo → mono

    all_windows = _build_windows(speech_segments)
    if not all_windows:
        logger.info("No windows to embed (all segments below MIN_SEGMENT_DURATION)")
        return np.empty((0, 256), dtype=np.float32), []

    # One WeSpeaker session per thread.  The underlying ONNX Runtime
    # session is not documented as thread-safe, and creating a fresh
    # Speaker is cheap (~26 MB model, one-time load per worker).
    _tls = threading.local()

    def _get_speaker():
        sp = getattr(_tls, "speaker", None)
        if sp is None:
            sp = wespeaker_rt.Speaker(lang="en")
            _tls.speaker = sp
        return sp

    def _process(item: tuple[float, float, int]):
        win_start, win_end, parent_idx = item
        start_sample = int(win_start * sr)
        end_sample = int(win_end * sr)
        segment_audio = audio_data[start_sample:end_sample]

        tmp_path: str | None = None
        try:
            # wespeakerruntime accepts file paths — write segment to a temp wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, segment_audio, sr)
            emb = _get_speaker().extract_embedding(tmp_path)
        except Exception:
            logger.debug(
                "Embedding extraction failed for window %.2f-%.2f",
                win_start,
                win_end,
            )
            return None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if emb is None:
            return None
        if emb.ndim == 2:
            emb = emb[0]
        return emb, SubSegment(start=win_start, end=win_end, parent_idx=parent_idx)

    embeddings: list[np.ndarray] = []
    subsegments: list[SubSegment] = []

    total = len(all_windows)
    # Emit roughly 30 progress updates over the whole run, but never more
    # often than every PROGRESS_EVERY windows. Short runs therefore stay
    # quiet; long runs get a steady stream without flooding the SSE pipe.
    stride = max(PROGRESS_EVERY, total // 30) if total > 0 else PROGRESS_EVERY

    def _handle_result(r, done: int) -> None:
        if r is not None:
            embeddings.append(r[0])
            subsegments.append(r[1])
        if done % stride == 0 or done == total:
            logger.info("embedding %d/%d", done, total)

    if workers <= 1:
        # Serial path — matches upstream behaviour exactly.
        for done, item in enumerate(all_windows, 1):
            _handle_result(_process(item), done)
    else:
        # ThreadPoolExecutor.map preserves submission order, so the
        # resulting (embeddings, subsegments) lists are positionally
        # identical to the serial version.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for done, r in enumerate(pool.map(_process, all_windows), 1):
                _handle_result(r, done)

    if not embeddings:
        return np.empty((0, 256), dtype=np.float32), []

    X = np.stack(embeddings)  # (N, 256)
    logger.info("Extracted %d embeddings (dim=%d)", X.shape[0], X.shape[1])
    return X, subsegments