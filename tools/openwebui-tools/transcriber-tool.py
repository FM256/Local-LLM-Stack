"""
title: Transcriber
description: This tool allows to transcribe the content of video and audio files, as well as Youtube Video URLs (such as youtube.com/watch?v=... or youtu.be/...). The tool returns the full transcript as plain text. It handles audio and video ONLY — do NOT use it for images, screenshots, or diagrams; analyze those directly with your built-in vision capabilities instead.
author: FM256
requirements: httpx
"""

import base64
import json
import os
import re
import time
from pathlib import Path

import httpx

# Extensions and magic-byte signatures used to reject image uploads early.
# The transcriber only handles audio and video; images should be analyzed
# with the assistant's built-in vision, not sent through this tool.
_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
    ".svg",
}


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as '1h 2m 3s'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _looks_like_image(filename: str, data: bytes) -> str | None:
    """Return a short image-format label if `filename`/`data` look like an
    image, else None. Checks extension first, then magic bytes."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in _IMAGE_EXTS:
        return ext.lstrip(".").upper()
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WebP"
    if len(data) >= 2 and data[:2] == b"BM":
        return "BMP"
    if len(data) >= 4 and data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if (
        len(data) >= 12
        and data[4:8] == b"ftyp"
        and data[8:12]
        in (
            b"heic",
            b"heix",
            b"hevc",
            b"hevx",
            b"mif1",
            b"msf1",
            b"avif",
        )
    ):
        return "HEIF/HEIC/AVIF"
    return None


# Whitelist of prefixes that are allowed through to the user. Anything
# not starting with one of these (or matching the transcription /
# diarization progress patterns handled below) is dropped. This is a
# whitelist, not a blacklist — chatter from the transcribe CLI or the
# diarize library must not leak into the Open WebUI event log.
#
#   📥  download phase (transcribe CLI)
#   📤  upload phase (this tool)
#   🎵  audio extraction / conversion
#   🎧  transcribing header
#   🎙️  diarization phase
#   📝  transcription progress and final "ready" line
#   ✅  phase completion markers
#   ⚠️  warnings
#   ❌  errors
_FORWARD_PREFIXES = ("📥", "📤", "🎵", "🎧", "🎙️", "📝", "✅", "⚠️", "❌")

# Match whisper-cli segment lines: "[HH:MM:SS.mmm --> HH:MM:SS.mmm] text"
_SEGMENT_RE = re.compile(r"^\[\d+:\d+:\d+[.,]\d+\s*(?:-->|→)\s*" r"(\d+):(\d+):(\d+)[.,](\d+)\]")

# Match diarize's embedding counter: "embedding N/M"
_EMBEDDING_RE = re.compile(r"^embedding (\d+)/(\d+)\s*$")

# Match the audio-duration hint: "🎵 Audio duration: 1h 2m 26s (3746.0s)"
_DURATION_RE = re.compile(r"^🎵 Audio duration:.*?\(([\d.]+)s\)")


class Tools:
    def __init__(self):
        self.tool_server_base = os.getenv("TOOL_SERVER_URL", "http://llm-tools:8000")
        self.url_endpoint = self.tool_server_base + "/transcribe"
        self.file_endpoint = self.tool_server_base + "/transcribe_file"
        # Timeout in seconds - configurable via env var, default 2 hours.
        # The server heartbeats during silent phases so this should rarely
        # fire, but it needs headroom above the server's per-request timeout.
        self.timeout = float(os.getenv("TOOL_TIMEOUT", "7200.0"))

    @staticmethod
    def _coarsen_progress(msg: str, state: dict) -> str | None:
        """Filter a raw progress line from the server.

        The server emits very fine-grained events (one per whisper
        segment, one per embedding batch) which are useful for a live
        terminal but turn the Open WebUI event log into noise. This
        translates the stream into coarse phase + 10%-step updates:

          - phase transitions (🎵 / 🎧 / 🎙️ / ✅ / ⚠️ / ❌) pass through
          - segment lines are folded into "Transcription N%" updates
            emitted at 10% boundaries only
          - embedding counters fold into "Diarization N%" at 10%
          - heartbeats (⏳) are dropped entirely
          - everything else passes through unchanged

        Returns the message to forward, or None to suppress.
        """
        # Audio-duration hint: consume it, do not display.
        m = _DURATION_RE.match(msg)
        if m:
            state["duration"] = float(m.group(1))
            return None

        # Heartbeat: suppress.
        if msg.startswith("⏳"):
            return None

        # Segment line: compute transcript percent, forward at 10% steps.
        m = _SEGMENT_RE.match(msg)
        if m:
            dur = state.get("duration")
            if dur and dur > 0:
                h, mm, ss, ms = (int(x) for x in m.groups())
                end = h * 3600 + mm * 60 + ss + ms / 1000.0
                pct = max(0, min(100, int(end / dur * 100)))
                bucket = pct // 10 * 10
                if bucket > state["trans_pct"]:
                    state["trans_pct"] = bucket
                    if bucket > 0:
                        return f"📝 Transcription {bucket}%"
            return None

        # Embedding counter: forward at 10% steps.
        m = _EMBEDDING_RE.match(msg)
        if m:
            n, total = int(m.group(1)), int(m.group(2))
            if total > 0:
                pct = int(n / total * 100)
                bucket = pct // 10 * 10
                if bucket > state["diar_pct"]:
                    state["diar_pct"] = bucket
                    if bucket > 0:
                        return f"🎙️ Diarization {bucket}%"
            return None

        # Whitelisted prefix: forward.
        if msg.startswith(_FORWARD_PREFIXES):
            return msg

        # Anything else: drop. This is where CLI chatter and library
        # log noise ("Diarizing: tmp...", "Running Voice Activity
        # Detection...", "VAD complete: ...", "Clustering with fixed
        # num_speakers=2", "Diarization complete: ...") goes.
        return None

    async def _process_sse(self, response, __event_emitter__):
        """
        Common SSE handler for both YouTube and file transcription.
        Accumulates data lines for result/error events; emits coarsened
        progress (phases + 10% checkpoints) via __event_emitter__.

        Returns ``(transcript, state)`` where ``state`` is the progress
        filter's state dict. Callers use ``state["duration"]`` to report
        audio length in the final status line.
        """
        current_event = None
        event_data = []

        # Progress-filter state, scoped to one SSE session:
        #   duration   — total audio duration in seconds, learned from
        #                the server's "🎵 Audio duration" hint
        #   trans_pct  — last transcription bucket we emitted (0/10/.../100)
        #   diar_pct   — last diarization bucket we emitted
        progress_state = {"duration": None, "trans_pct": -10, "diar_pct": -10}

        async for line in response.aiter_lines():
            if line == "":
                # Event ended – process accumulated data
                if current_event == "result":
                    result = "\n".join(event_data)
                    if not result.strip():
                        raise RuntimeError("Server returned an empty transcript.")
                    return result, progress_state
                elif current_event == "error":
                    raise RuntimeError("\n".join(event_data))
                elif current_event == "progress":
                    # already handled per line, ignore
                    pass
                # Reset for next event
                current_event = None
                event_data = []
                continue

            if line.startswith("event:"):
                # In case we had an event that wasn't terminated, flush it
                if current_event == "result" and event_data:
                    result = "\n".join(event_data)
                    if not result.strip():
                        raise RuntimeError("Server returned an empty transcript.")
                    return result, progress_state
                elif current_event == "error" and event_data:
                    raise RuntimeError("\n".join(event_data))
                current_event = line.split(":", 1)[1].strip()
                event_data = []
                continue

            if line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
                if current_event == "progress":
                    if __event_emitter__:
                        coarsened = self._coarsen_progress(data, progress_state)
                        if coarsened:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"description": coarsened, "done": False},
                                }
                            )
                else:
                    # accumulate for result/error
                    event_data.append(data)
                continue

        # End of stream – process remaining event
        if current_event == "result" and event_data:
            result = "\n".join(event_data)
            if not result.strip():
                raise RuntimeError("Server returned an empty transcript.")
            return result, progress_state
        elif current_event == "error" and event_data:
            raise RuntimeError("\n".join(event_data))
        raise RuntimeError(
            "No transcript received from server (stream ended without result or error)."
        )

    async def transcribe_youtube(
        self,
        url: str,
        diarize: bool = False,
        num_speakers: int = None,
        __event_emitter__=None,
    ) -> str:
        """
        Generates the full plain-text transcript of a YouTube video.

        Use this tool ONLY when:
        - The user explicitly asks for the transcript, text content, or written summary of a YouTube video.
        - The user asks you to read, analyze, summarize or answer a question about the *content* of a YouTube video.

        :param url: The full YouTube URL (e.g., "https://www.youtube.com/watch?v=dQw4w9WgXcQ" or "https://youtu.be/dQw4w9WgXcQ").
        :param diarize: Set to True ONLY when the user explicitly asks for speaker diarization,
            speaker labels, "who said what", or to distinguish/identify different speakers.
            Leave as False (the default) in every other case.
        :param num_speakers: Optional number of speakers (only used when diarize=True). If the user
            explicitly states a speaker count, set it here; otherwise leave as None for auto-detection.
        :param __event_emitter__: Event emitter (injected by Open WebUI)
        :return: The full transcript as a string, or an error message.
        """
        start_time = time.monotonic()

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"📝 Starting transcription for {url}...",
                        "done": False,
                    },
                }
            )

        payload = {"url": url, "diarize": diarize}
        if num_speakers is not None:
            payload["num_speakers"] = num_speakers

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.url_endpoint, json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise RuntimeError(f"Server returned {response.status_code}: {error_text}")

                    transcript, sse_state = await self._process_sse(response, __event_emitter__)

                    if __event_emitter__:
                        # The human-readable summary ("✅ Transcribed ...")
                        # is emitted by transcribe's ProgressReporter and
                        # reaches us as a whitelisted progress event, so we
                        # do not duplicate it here. This empty-desc status
                        # exists only to signal task completion to Open WebUI.
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": "", "done": True},
                            }
                        )
                    return transcript

        except httpx.TimeoutException as e:
            error_msg = (
                f"Transcription timed out after {int(self.timeout)} seconds. "
                "The video may be too long or the server is slow. "
                f"Original error: {e}"
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg) from e

        except httpx.HTTPError as e:
            error_msg = f"HTTP error during transcription: {e}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg) from e

    async def transcribe_file(
        self,
        file_index: int,
        diarize: bool = False,
        num_speakers: int = None,
        __files__=None,
        __event_emitter__=None,
    ) -> str:
        """
        Generates the full plain-text transcript of an attached audio or video file.

        Use this tool ONLY when:
        - The user attached one or more media files (audio/video) to their prompt.
        - The user asks you to transcribe, analyze, summarize, or answer questions about the *content* of a specific attached file.

        Do NOT use this tool for images, screenshots, or PDFs. This tool handles
        audio and video files ONLY.

        HOW TO CHOOSE THE RIGHT FILE INDEX:
        - Use 0‑based indexing (0 = first, 1 = second, etc.). Indexes count all attached files across the whole conversation, not just the current turn. Make sure to count attachments from previous chat turns when determining the index to use.
        - If you want to transcribe multiple files, call this tool once for each file that needs transcribing.

        :param file_index: The 0‑based index of the attached file to transcribe.
        :param diarize: Set to True ONLY when the user explicitly asks for speaker diarization,
            speaker labels, "who said what", or to distinguish/identify different speakers.
            Leave as False (the default) in every other case.
        :param num_speakers: Optional number of speakers (only used when diarize=True). If the user
            explicitly states a speaker count, set it here; otherwise leave as None for auto-detection.
        :param __files__: List of attached files (injected by OpenWebUI).
        :param __event_emitter__: Event emitter (injected by OpenWebUI).
        :return: The full transcript as a string, or an error message.
        """
        start_time = time.monotonic()

        # --- Validate input ---
        if not __files__:
            error_msg = "No file attached. Please attach an audio or video file."
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg)

        if file_index < 0 or file_index >= len(__files__):
            error_msg = (
                f"File index {file_index} is out of range. You attached {len(__files__)} file(s). "
                f"Valid indices are 0 to {len(__files__)-1}."
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg)

        # --- Get the selected file ---
        file_info = __files__[file_index]
        filename = file_info.get("name", "unknown")

        # --- Extract file bytes (robust) ---
        file_bytes = None

        # Attempt 1: Inline binary data (fresh upload)
        try:
            content = file_info["file"]["data"]["content"]
            if isinstance(content, str):
                file_bytes = base64.b64decode(content)
            else:
                file_bytes = content
        except (KeyError, TypeError):
            # Attempt 2: Read from filesystem using top-level id/filename
            try:
                file_id = file_info.get("id") or file_info.get("file", {}).get("id")
                file_name = (
                    file_info.get("filename")
                    or file_info.get("file", {}).get("filename")
                    or file_info.get("name")
                )
                if not file_id or not file_name:
                    raise RuntimeError("No file ID or filename found in __files__ entry.")
                file_path = Path(f"/app/backend/data/uploads/{file_id}_{file_name}")
                if not file_path.exists():
                    alt_path = Path(f"/app/backend/data/uploads/{file_id}")
                    if alt_path.exists():
                        file_path = alt_path
                    else:
                        raise RuntimeError(f"File not found at {file_path} or {alt_path}")
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
            except Exception as e:
                error_msg = f"Failed to access file data: {e}"
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": f"❌ {error_msg}", "done": True},
                        }
                    )
                raise RuntimeError(error_msg)

        if file_bytes is None:
            raise RuntimeError("Could not obtain file bytes from any source.")

        if len(file_bytes) < 100:
            error_msg = f"File content is too small ({len(file_bytes)} bytes) – may be corrupted."
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg)

        # --- Reject images early: the assistant has vision, this tool does not ---
        image_kind = _looks_like_image(filename, file_bytes)
        if image_kind:
            error_msg = (
                f"'{filename}' looks like a {image_kind} image, not audio or "
                "video. This tool only transcribes audio and video. Do not "
                "call it for images or screenshots — analyze them directly "
                "with your built-in vision capabilities instead."
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg)

        # --- Emit status and send to backend ---
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"📤 Uploading and transcribing '{filename}' (index {file_index}, {len(file_bytes)} bytes)...",
                        "done": False,
                    },
                }
            )

        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = {"diarize": "true" if diarize else "false"}
        if num_speakers is not None:
            data["num_speakers"] = str(num_speakers)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", self.file_endpoint, files=files, data=data
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise RuntimeError(f"Server returned {response.status_code}: {error_text}")

                    transcript, sse_state = await self._process_sse(response, __event_emitter__)

                    if __event_emitter__:
                        # The human-readable summary ("✅ Transcribed ...")
                        # is emitted by transcribe's ProgressReporter and
                        # reaches us as a whitelisted progress event, so we
                        # do not duplicate it here. This empty-desc status
                        # exists only to signal task completion to Open WebUI.
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": "", "done": True},
                            }
                        )
                    return transcript

        except httpx.TimeoutException as e:
            error_msg = (
                f"Transcription timed out after {int(self.timeout)} seconds. "
                f"Original error: {e}"
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg) from e

        except httpx.HTTPError as e:
            error_msg = f"HTTP error during transcription: {e}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": f"❌ {error_msg}", "done": True},
                    }
                )
            raise RuntimeError(error_msg) from e
