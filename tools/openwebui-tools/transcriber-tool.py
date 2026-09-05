"""
title: Transcriber
description: This tool allows to transcribe the content of video and audio files, as well as Youtube Video URLs (such as youtube.com/watch?v=... or youtu.be/...). The tool returns the full transcript as plain text.
author: FM256
requirements: httpx
"""

import os
import httpx
import json
import base64
from pathlib import Path


class Tools:
    def __init__(self):
        self.tool_server_base = os.getenv("TOOL_SERVER_URL", "http://llm-tools:8000")
        self.url_endpoint = self.tool_server_base + "/transcribe"
        self.file_endpoint = self.tool_server_base + "/transcribe_file"
        # Timeout in seconds - configurable via env var, default 20 minutes
        self.timeout = float(os.getenv("TOOL_TIMEOUT", "1200.0"))

    async def _process_sse(self, response, __event_emitter__):
        """
        Common SSE handler for both YouTube and file transcription.
        Accumulates data lines for result/error events; emits progress immediately.
        Returns the final transcript or raises RuntimeError with full error details.
        """
        current_event = None
        event_data = []

        async for line in response.aiter_lines():
            if line == "":
                # Event ended – process accumulated data
                if current_event == "result":
                    result = "\n".join(event_data)
                    if not result.strip():
                        raise RuntimeError("Server returned an empty transcript.")
                    return result
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
                    return result
                elif current_event == "error" and event_data:
                    raise RuntimeError("\n".join(event_data))
                current_event = line.split(":", 1)[1].strip()
                event_data = []
                continue

            if line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
                if current_event == "progress":
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": f"📝 {data}", "done": False},
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
            return result
        elif current_event == "error" and event_data:
            raise RuntimeError("\n".join(event_data))
        raise RuntimeError(
            "No transcript received from server (stream ended without result or error)."
        )

    async def transcribe_youtube(self, url: str, __event_emitter__=None) -> str:
        """
        Generates the full plain-text transcript of a YouTube video.

        Use this tool ONLY when:
        - The user explicitly asks for the transcript, text content, or written summary of a YouTube video.
        - The user asks you to read, analyze, summarize or answer a question about the *content* of a YouTube video.

        :param url: The full YouTube URL (e.g., "https://www.youtube.com/watch?v=dQw4w9WgXcQ" or "https://youtu.be/dQw4w9WgXcQ").
        :param __event_emitter__: Event emitter (injected by Open WebUI)
        :return: The full transcript as a string, or an error message.
        """
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

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.url_endpoint, json={"url": url}) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise RuntimeError(f"Server returned {response.status_code}: {error_text}")

                    transcript = await self._process_sse(response, __event_emitter__)

                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "📝 Transcript ready (YouTube).",
                                    "done": True,
                                },
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

    async def transcribe_file(self, file_index: int, __files__=None, __event_emitter__=None) -> str:
        """
        Generates the full plain-text transcript of an attached audio or video file.

        Use this tool ONLY when:
        - The user attached one or more media files (audio/video) to their prompt.
        - The user asks you to transcribe, analyze, summarize, or answer questions about the *content* of a specific attached file.

        HOW TO CHOOSE THE RIGHT FILE INDEX:
        - Use 0‑based indexing (0 = first, 1 = second, etc.). Indexes count all attached files across the whole conversation, not just the current turn. Make sure to count attachments from previous chat turns when determining the index to use.
        - If you want to transcribe multiple files, call this tool once for each file that needs transcribing.

        :param file_index: The 0‑based index of the attached file to transcribe.
        :param __files__: List of attached files (injected by OpenWebUI).
        :param __event_emitter__: Event emitter (injected by OpenWebUI).
        :return: The full transcript as a string, or an error message.
        """
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

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", self.file_endpoint, files=files) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise RuntimeError(f"Server returned {response.status_code}: {error_text}")

                    transcript = await self._process_sse(response, __event_emitter__)

                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "📝 Transcript ready (file).",
                                    "done": True,
                                },
                            }
                        )
                    return transcript

        except httpx.TimeoutException as e:
            error_msg = (
                f"Transcription timed out after {int(self.timeout)} seconds. "
                "The file may be too large or the server is slow. "
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
