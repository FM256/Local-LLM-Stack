import asyncio
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from diarize import diarize
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

# ----------------------------------------------------------------------
# Logging and Debug Flag
# ----------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
if DEBUG:
    logger.info("🔍 DEBUG mode enabled – verbose logs will be printed")

app = FastAPI()


def _sse(event: str, data: str) -> str:
    """Format a Server-Sent Event. Multi-line data is split into one
    'data:' line per source line, per the SSE spec."""
    out = [f"event: {event}"]
    for line in (data or "").split("\n"):
        out.append(f"data: {line}")
    out.append("")
    out.append("")
    return "\n".join(out)

# ----------------------------------------------------------------------
# Constants & paths
# ----------------------------------------------------------------------
WHISPER_SERVER_URL = "http://localhost:8080/inference"
WHISPER_CLI = "/app/build/bin/whisper-cli"
MODEL_PATH = "/app/models/ggml-small.bin"
VAD_MODEL_PATH = "/app/models/ggml-silero-v5.1.2.bin"
FFMPEG = "/usr/bin/ffmpeg"
# These are the only formats supported by Whisper and Whisper CLI! Others must be reencoded accordingly.
# Note that .wav is the native format whisper actually works with, so using this is the most
# efficient, as otherwise it will re-encode internally, anyways
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}

# Anti-repetition / anti-hallucination flags for whisper-cli.
#
# The repetition-loop failure mode (one phrase emitted many times in a
# row) is driven by decoder context carryover: once the model repeats,
# the repeated text becomes context, raising the probability of another
# repeat. `-mc 0` removes context carryover entirely, breaking the
# feedback loop at the root. `-sns` suppresses non-speech tokens such
# as [BLANK_AUDIO] and [MUSIC], which are themselves a frequent trigger.
#
# -et / -lpt / -nth are listed explicitly even though they currently
# match upstream defaults, so we have one place to tune them if the A/B
# run shows we need to be more aggressive.
WHISPER_ANTI_REPETITION = [
    "-mc", "0",
    "-sns",
    "-et", "2.4",
    "-lpt", "-1.0",
    "-nth", "0.6",
]

# Silero VAD pre-filtering. Whisper-cli segments the audio by speech
# activity before decoding, so silent stretches never reach the decoder.
# Silence is the primary trigger for hallucination loops: with no speech
# to attend to, the model emits the highest-probability sequence, which
# becomes context and self-reinforces. -mc 0 breaks that reinforcement
# *after* a repeat begins; VAD prevents the first one.
#
# The model must be the GGML-converted Silero VAD weights — same model
# the diarize library uses on the Python side, but transcoded into a
# format whisper.cpp can load.
WHISPER_VAD = [
    "--vad",
    "--vad-model", VAD_MODEL_PATH,
]


# ----------------------------------------------------------------------
# Diarization helpers
# ----------------------------------------------------------------------
def run_diarization_and_merge(
    wav_path: str, transcription_segments: List[Dict[str, Any]], num_speakers: Optional[int] = None
) -> List[Dict[str, Any]]:
    if not transcription_segments:
        return transcription_segments

    logger.info(f"Running diarization on {wav_path} with num_speakers={num_speakers}")

    # Run diarize with forced min/max speakers if num_speakers is provided
    try:
        result = diarize(wav_path, num_speakers=num_speakers)
    except Exception as e:
        logger.exception("Diarization library failed")
        return transcription_segments

    # Debug: log raw diarization segments
    if DEBUG and result.segments:
        logger.debug(
            f"Raw diarization segments: {[(seg.start, seg.end, seg.speaker) for seg in result.segments]}"
        )

    # Build speaker segments
    diar_segments = [
        {"start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in result.segments
    ]

    # Merge with transcription segments by overlap (same logic as before)
    merged = []
    for t_seg in transcription_segments:
        best_overlap = 0.0
        best_speaker = "SPEAKER_00"
        for d_seg in diar_segments:
            overlap_start = max(t_seg["start"], d_seg["start"])
            overlap_end = min(t_seg["end"], d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]
        merged_seg = t_seg.copy()
        merged_seg["speaker"] = best_speaker
        merged.append(merged_seg)

    logger.info(f"Assigned speakers to {len(merged)} transcription segments.")
    return merged


# ----------------------------------------------------------------------
# Utility functions (timestamp parsing, WAV conversion, OpenAI formatting)
# ----------------------------------------------------------------------
def parse_timestamp(ts_str: str) -> float:
    """
    Converts a whisper‑style timestamp string (e.g., "00:00:00,000") to seconds.

    Args:
        ts_str: Timestamp in the format "HH:MM:SS,mmm".

    Returns:
        float: The timestamp in seconds (with milliseconds as fractional part).
            Returns 0.0 if the string cannot be parsed.
    """
    parts = ts_str.split(",")
    if len(parts) != 2:
        return 0.0
    h, m, s = map(int, parts[0].split(":"))
    ms = int(parts[1])
    return h * 3600 + m * 60 + s + ms / 1000.0


def transform_to_openai_verbose(
    text: str, segments: List[Dict], language: Optional[str] = None
) -> Dict:
    """
    Formats transcription results into the OpenAI‑compatible verbose JSON structure.

    The output includes a top‑level "text", "language", "duration", and a "segments"
    list. Each segment contains standard fields (id, start, end, text, tokens, etc.)
    and any additional fields (like "speaker") if present in the input segments.

    Args:
        text: The full transcribed text.
        segments: A list of segment dictionaries. Each segment must contain
            "start", "end", and "text" keys. It may also contain a "speaker" key.
        language: The detected language (or "auto"). If None, defaults to "auto".

    Returns:
        Dict: A dictionary conforming to the OpenAI Whisper verbose JSON format.
    """
    duration = segments[-1]["end"] if segments else 0.0
    openai_segments = []
    for idx, seg in enumerate(segments):
        openai_segments.append(
            {
                "id": idx,
                "seek": 0,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
        )
        if "speaker" in seg:
            openai_segments[-1]["speaker"] = seg["speaker"]
    return {
        "text": text,
        "language": language or "auto",
        "duration": duration,
        "segments": openai_segments,
    }


def _group_by_speaker(segments: List[Dict[str, Any]]) -> List[tuple]:
    """Merge consecutive segments belonging to the same speaker.

    Used only for plain-text output, where a run of same-speaker
    segments reads better as a single utterance than as a stack of
    repeated speaker labels. JSON/SRT/VTT output is unaffected — those
    formats legitimately need per-segment records.

    Empty-text segments are dropped. Returns a list of
    ``(speaker, text)`` pairs in original order.
    """
    grouped: List[tuple] = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if grouped and grouped[-1][0] == speaker:
            prev_speaker, prev_text = grouped[-1]
            grouped[-1] = (prev_speaker, f"{prev_text} {text}")
        else:
            grouped.append((speaker, text))
    return grouped


async def convert_to_wav(input_path: str) -> str:
    """
    Converts any audio file to a 16 kHz mono WAV using ffmpeg. This is the native format whisper
    will want to work with.

    Args:
        input_path: Path to the input audio file.

    Returns:
        str: Path to the newly created temporary WAV file.

    Raises:
        HTTPException: If ffmpeg conversion fails (status 400).
    """
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    cmd = [
        FFMPEG,
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-y",
        wav_path,
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        os.unlink(wav_path)
        raise HTTPException(status_code=400, detail=f"FFmpeg conversion failed: {stderr.decode()}")
    return wav_path


# ----------------------------------------------------------------------
# FastAPI endpoints
# ----------------------------------------------------------------------
@app.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
        JSONResponse: {"status": "ok"}.
    """
    return {"status": "ok"}


@app.get("/backend")
async def get_backend():
    """
    Reports which backend (CPU/ROCm) whisper.cpp is using.

    Returns:
        JSONResponse: {"backend": <value from WHISPER_BACKEND env var>}
    """
    backend = os.environ.get("WHISPER_BACKEND", "unknown")
    return {"backend": backend}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    collapse: str = Form("true"),
    diarize: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
):
    """
    OpenAI‑compatible audio transcription endpoint with optional speaker diarization.

    This endpoint accepts an audio file and returns a transcription in the format
    requested (json, text, srt, vtt, or verbose_json). When `diarize` is True,
    speaker labels are added to each segment.

    Args:
        request: The FastAPI Request object (used to detect client disconnection).
        file: The uploaded audio file.
        model: The model name (ignored, but kept for OpenAI compatibility).
        language: Optional language code to force (e.g., "de"). If None, auto‑detection is used.
        response_format: Output format: "json", "text", "srt", "vtt", or "verbose_json".
        temperature: Temperature parameter for the Whisper model (only used when
            calling the whisper‑server).
        collapse: If "true" (default), collapses whitespace in text output.
        diarize: If True, runs speaker diarization and adds "speaker" labels.
        num_speakers: Optional number of speakers for diarization. If None, auto‑detect.

    Returns:
        Response: Depending on format:
            - JSON (JSONResponse): OpenAI‑compatible verbose JSON (with segments).
            - Text (plain text): The plain transcription text.
            - SRT/VTT (plain text): Subtitle file contents.

    Raises:
        HTTPException: On client disconnection (499), ffmpeg errors (400),
                       whisper‑cli/server errors (500), or unsupported formats.
    """
    logger.info("=" * 80)
    logger.info("NEW REQUEST")
    logger.info(f"  file.filename: {file.filename}")
    logger.info(f"  model: {model}")
    logger.info(f"  language: {language}")
    logger.info(f"  response_format: {response_format}")
    logger.info(f"  temperature: {temperature}")
    logger.info(f"  collapse: {collapse}")
    logger.info(f"  diarize: {diarize}")
    logger.info(f"  num_speakers: {num_speakers}")
    logger.info("=" * 80)

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    original_filename = file.filename or "audio"
    ext = os.path.splitext(original_filename)[1].lower()

    # Convert to WAV if needed
    if ext not in SUPPORTED_EXTENSIONS:
        logger.info(f"Unsupported format '{ext}', converting to WAV...")
        try:
            process_path = await convert_to_wav(tmp_path)
        except HTTPException as e:
            os.unlink(tmp_path)
            raise e
        os.unlink(tmp_path)
    else:
        new_path = tmp_path + ext
        os.rename(tmp_path, new_path)
        process_path = new_path

    process = None
    try:
        # ------------------------------------------------------------------
        # 1) Formats handled by whisper-cli:
        #    - srt, vtt, verbose_json (native)
        #    - json and text when diarize is True (we need segments)
        # ------------------------------------------------------------------
        use_cli = response_format in ("srt", "vtt", "verbose_json") or (
            response_format in ("json", "text") and diarize
        )

        if use_cli:
            if DEBUG:
                logger.debug("Using whisper-cli (because diarize is true or native verbose format)")

            base_out = process_path
            # whisper-cli defaults to English if -l is omitted, which
            # silently translates non-English audio. Always pass -l,
            # falling back to 'auto' so detection actually runs.
            cmd = [
                WHISPER_CLI, "-m", MODEL_PATH, "-f", process_path,
                "-l", language or "auto",
                *WHISPER_VAD, *WHISPER_ANTI_REPETITION,
            ]

            # Determine CLI output format
            if response_format == "srt":
                cmd += ["-osrt", "-of", base_out, "-np"]
            elif response_format == "vtt":
                cmd += ["-ovtt", "-of", base_out, "-np"]
            else:
                # For verbose_json, json, or text (with diarize) we need JSON output
                cmd += ["-oj", "-of", base_out, "-np"]

            if DEBUG:
                logger.debug(f"Running whisper-cli: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            communicate_task = asyncio.create_task(process.communicate())

            while not communicate_task.done():
                if await request.is_disconnected():
                    logger.warning("Client disconnected – killing whisper-cli")
                    if process.returncode is None:
                        process.kill()
                    communicate_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                try:
                    await asyncio.wait_for(asyncio.shield(communicate_task), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    if not communicate_task.done():
                        communicate_task.cancel()
                    break

            stdout, stderr = await communicate_task
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"CLI failed: {stderr.decode()}")

            # Read output file (either .srt, .vtt, or .json)
            if response_format == "srt":
                output_file = base_out + ".srt"
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        raw_output = f.read()
                    return Response(content=raw_output, media_type="text/plain")
                else:
                    raise HTTPException(status_code=500, detail="SRT file not produced")
            elif response_format == "vtt":
                output_file = base_out + ".vtt"
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        raw_output = f.read()
                    return Response(content=raw_output, media_type="text/plain")
                else:
                    raise HTTPException(status_code=500, detail="VTT file not produced")
            else:
                # json or verbose_json: read .json
                output_file = base_out + ".json"
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        raw_output = f.read()
                else:
                    raw_output = stdout.decode()  # fallback

                # Parse JSON from CLI
                try:
                    cli_data = json.loads(raw_output)
                except json.JSONDecodeError as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid JSON from whisper-cli: {e}\nRaw output: {raw_output[:200]}",
                    )

                detected_lang = cli_data.get("result", {}).get("language", "auto")
                transcription_entries = cli_data.get("transcription", [])
                segments = []
                full_text_parts = []
                for entry in transcription_entries:
                    ts = entry.get("timestamps", {})
                    start_str = ts.get("from", "00:00:00,000")
                    end_str = ts.get("to", "00:00:00,000")
                    start = parse_timestamp(start_str)
                    end = parse_timestamp(end_str)
                    text = entry.get("text", "").strip()
                    if text:
                        segments.append({"start": start, "end": end, "text": text})
                        full_text_parts.append(text)

                full_text = " ".join(full_text_parts)
                if not segments and "text" in cli_data:
                    full_text = cli_data["text"]
                    segments = [{"start": 0.0, "end": 0.0, "text": full_text}]

                # Apply diarization if requested
                if diarize and segments:
                    try:
                        segments = run_diarization_and_merge(process_path, segments, num_speakers)
                    except Exception as e:
                        logger.exception("Diarization failed, returning without speaker labels")

                # Build OpenAI-compatible verbose JSON
                openai_result = transform_to_openai_verbose(full_text, segments, detected_lang)

                # ----------------------------------------------------------
                # Handle plain text with speaker labels (if response_format is "text")
                # ----------------------------------------------------------
                if response_format == "text" and diarize and segments:
                    collapse_bool = collapse.lower() in ("true", "1", "yes")
                    lines = []
                    for speaker, text in _group_by_speaker(segments):
                        if collapse_bool:
                            text = " ".join(text.split())
                        lines.append(f"{speaker}:\n{text}")
                    # Join merged speaker turns with a blank line between them
                    output_text = "\n\n".join(lines)
                    return Response(content=output_text, media_type="text/plain")

                # If the user requested "json" or "verbose_json", return JSON
                if response_format == "json" or response_format == "verbose_json":
                    return JSONResponse(content=openai_result)
                else:
                    # This case shouldn't happen, but fallback to plain text
                    return Response(content=openai_result["text"], media_type="text/plain")

        # ------------------------------------------------------------------
        # 2) Formats handled by whisper-server: only json when diarize=False
        #    (and also text without diarization)
        # ------------------------------------------------------------------
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        with open(process_path, "rb") as f:
            files = {"file": (os.path.basename(process_path), f, "audio/mpeg")}
            # When diarize is False, we can use simple json
            request_format = "json"
            data = {"response-format": request_format}
            if temperature is not None:
                data["temperature"] = str(temperature)
            if language is not None:
                data["language"] = language

            if DEBUG:
                logger.debug(f"📤 Sending to whisper-server: URL={WHISPER_SERVER_URL}, data={data}")

            async with httpx.AsyncClient(timeout=600.0) as client:
                send_task = asyncio.create_task(
                    client.post(WHISPER_SERVER_URL, files=files, data=data)
                )
                while not send_task.done():
                    if await request.is_disconnected():
                        logger.warning("Client disconnected while waiting for whisper-server")
                        send_task.cancel()
                        raise HTTPException(status_code=499, detail="Client disconnected")
                    try:
                        await asyncio.wait_for(asyncio.shield(send_task), timeout=0.1)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                resp = await send_task

        if DEBUG:
            logger.debug(f"📥 Response status: {resp.status_code}")
            body_preview = resp.text[:500] + ("..." if len(resp.text) > 500 else "")
            logger.debug(f"   Response body preview: {body_preview}")

        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Whisper server error")

        try:
            server_data = resp.json()
        except Exception as e:
            if DEBUG:
                logger.debug(f"Failed to parse JSON, treating as plain text: {e}")
            server_data = {"text": resp.text.strip()}

        if DEBUG:
            logger.debug(f"📦 Parsed server_data keys: {list(server_data.keys())}")

        if response_format == "json":
            # For json response, we return whatever the server gave (no diarization because diarize is False here)
            return JSONResponse(content=server_data)
        else:
            # "text" format without diarization
            transcription = server_data.get("text", resp.text.strip())
            if collapse.lower() in ("true", "1", "yes"):
                transcription = " ".join(transcription.split())
            return Response(content=transcription, media_type="text/plain")

    except HTTPException:
        raise
    except asyncio.CancelledError:
        logger.warning("Request task was cancelled")
        raise HTTPException(status_code=499, detail="Client disconnected")
    except Exception as e:
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if process and process.returncode is None:
            try:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except:
                pass
        for path in [tmp_path, process_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        for ext in (".srt", ".vtt", ".json"):
            base = os.path.splitext(process_path)[0] if process_path else ""
            if base and os.path.exists(base + ext):
                try:
                    os.unlink(base + ext)
                except:
                    pass


class _SSELogHandler(logging.Handler):
    """Logging handler that forwards records to an asyncio queue.

    Safe to use from any thread; the actual queue mutation happens on
    the target event loop via ``call_soon_threadsafe``.  Used to bridge
    log records from the (thread-based) diarization pipeline into the
    SSE progress stream.  See the ``_stream_cli`` diarization block for
    where this is installed and torn down.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: "asyncio.Queue[Any]"):
        super().__init__()
        self._loop = loop
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)
        except RuntimeError:
            # Loop closed — drop silently during interpreter shutdown.
            pass


@app.post("/v1/audio/transcriptions/stream")
async def transcribe_stream(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    collapse: str = Form("true"),
    diarize: bool = Form(False),
    num_speakers: Optional[int] = Form(None),
):
    """
    Streaming variant of /v1/audio/transcriptions.

    Returns Server-Sent Events:
      - event: progress  data: <line>       (many)
      - event: result    data: <payload>    (exactly one, on success)
      - event: error     data: <message>    (zero or one, on failure)

    Payload semantics match the non-streaming endpoint for the same
    response_format: JSON string for json/verbose_json, plain text for
    text, subtitle content for srt/vtt.
    """
    logger.info(
        f"NEW STREAM REQUEST: file={file.filename} fmt={response_format} "
        f"diarize={diarize} num_speakers={num_speakers}"
    )
    return StreamingResponse(
        _stream_transcribe(
            request, file, language, response_format, temperature,
            collapse, diarize, num_speakers,
        ),
        media_type="text/event-stream",
    )


async def _stream_transcribe(
    request, file, language, response_format, temperature,
    collapse, diarize, num_speakers,
):
    """Async generator producing SSE events for a streaming transcription."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    original_filename = file.filename or "audio"
    ext = os.path.splitext(original_filename)[1].lower()
    process_path = None
    try:
        if ext not in SUPPORTED_EXTENSIONS:
            yield _sse("progress", f"🎵 Converting '{ext or 'unknown'}' to WAV...")
            try:
                process_path = await convert_to_wav(tmp_path)
            except HTTPException as e:
                yield _sse("error", f"Conversion failed: {e.detail}")
                return
            os.unlink(tmp_path)
        else:
            process_path = tmp_path + ext
            os.rename(tmp_path, process_path)

        use_cli = response_format in ("srt", "vtt", "verbose_json") or (
            response_format in ("json", "text") and diarize
        )

        if use_cli:
            yield _sse("progress", "🎧 Preparing transcription (model load + VAD)...")
            async for ev in _stream_cli(
                request, process_path, language, response_format,
                diarize, num_speakers, collapse,
            ):
                yield ev
        else:
            async for ev in _stream_server(
                request, process_path, language, response_format,
                temperature, collapse,
            ):
                yield ev

    except asyncio.CancelledError:
        logger.warning("Stream request cancelled")
        raise
    except Exception as e:
        logger.exception("Streaming transcribe failed")
        yield _sse("error", f"Internal error: {e}")
    finally:
        for p in [tmp_path, process_path]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass
        if process_path:
            base = os.path.splitext(process_path)[0]
            for suffix in (".srt", ".vtt", ".json"):
                p = base + suffix
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass


async def _stream_cli(
    request, process_path, language, response_format,
    diarize, num_speakers, collapse,
):
    """whisper-cli + optional diarization, streaming progress as SSE."""
    base_out = process_path
    # whisper-cli defaults to English if -l is omitted, which silently
    # translates non-English audio. Always pass -l, falling back to
    # 'auto' so detection actually runs.
    cmd = [
        WHISPER_CLI, "-m", MODEL_PATH, "-f", process_path,
        "-l", language or "auto",
        *WHISPER_VAD, *WHISPER_ANTI_REPETITION,
    ]
    if response_format == "srt":
        cmd += ["-osrt", "-of", base_out]
    elif response_format == "vtt":
        cmd += ["-ovtt", "-of", base_out]
    else:
        cmd += ["-oj", "-of", base_out]

    logger.info(f"Streaming whisper-cli: {' '.join(cmd)}")
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout_queue: asyncio.Queue = asyncio.Queue()

    async def _pump_stdout():
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                await stdout_queue.put(line)
        finally:
            await stdout_queue.put(None)

    async def _drain_stderr() -> bytes:
        chunks = []
        while True:
            chunk = await process.stderr.readline()
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    stdout_task = asyncio.create_task(_pump_stdout())
    stderr_task = asyncio.create_task(_drain_stderr())

    try:
        while True:
            if await request.is_disconnected():
                process.kill()
                await process.wait()
                yield _sse("error", "Client disconnected")
                return
            try:
                line_bytes = await asyncio.wait_for(stdout_queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield _sse("progress", "⏳ (transcribing...)")
                continue
            if line_bytes is None:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if line:
                yield _sse("progress", line)
        await process.wait()
        stderr_data = await stderr_task
    finally:
        if process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
        if not stdout_task.done():
            stdout_task.cancel()

    if process.returncode != 0:
        err_msg = stderr_data.decode("utf-8", errors="replace")
        yield _sse("error", f"whisper-cli exited with {process.returncode}:\n{err_msg}")
        return

    if response_format == "srt":
        output_file = base_out + ".srt"
        if not os.path.exists(output_file):
            yield _sse("error", "SRT file not produced")
            return
        yield _sse("progress", "✅ Transcription done.")
        with open(output_file, "r") as f:
            yield _sse("result", f.read())
        return
    if response_format == "vtt":
        output_file = base_out + ".vtt"
        if not os.path.exists(output_file):
            yield _sse("error", "VTT file not produced")
            return
        yield _sse("progress", "✅ Transcription done.")
        with open(output_file, "r") as f:
            yield _sse("result", f.read())
        return

    output_file = base_out + ".json"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            raw_output = f.read()
    else:
        yield _sse("error", "whisper-cli did not produce JSON output")
        return

    try:
        cli_data = json.loads(raw_output)
    except json.JSONDecodeError as e:
        yield _sse("error", f"Invalid JSON from whisper-cli: {e}")
        return

    detected_lang = cli_data.get("result", {}).get("language", "auto")
    transcription_entries = cli_data.get("transcription", [])
    segments = []
    full_text_parts = []
    for entry in transcription_entries:
        ts = entry.get("timestamps", {})
        start_str = ts.get("from", "00:00:00,000")
        end_str = ts.get("to", "00:00:00,000")
        start = parse_timestamp(start_str)
        end = parse_timestamp(end_str)
        text = entry.get("text", "").strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
            full_text_parts.append(text)

    full_text = " ".join(full_text_parts)
    if not segments and "text" in cli_data:
        full_text = cli_data["text"]
        segments = [{"start": 0.0, "end": 0.0, "text": full_text}]

    yield _sse("progress", "✅ Transcription done.")

    if diarize and segments:
        yield _sse("progress", f"🎙️ Starting diarization ({len(segments)} segments)...")

        # Bridge the diarize library's logging into our SSE stream.  The
        # library emits one progress record every N embedding windows
        # (see PROGRESS_EVERY in diarize/embeddings.py), which is where
        # virtually all the wall-clock time goes.  Records go from worker
        # threads -> call_soon_threadsafe -> log_queue -> here.
        loop = asyncio.get_running_loop()
        log_queue: "asyncio.Queue[Any]" = asyncio.Queue()
        _SENTINEL = object()

        diar_handler = _SSELogHandler(loop, log_queue)
        diar_handler.setLevel(logging.INFO)
        diar_handler.setFormatter(logging.Formatter("%(message)s"))

        diar_logger = logging.getLogger("diarize")
        prior_level = diar_logger.level
        diar_logger.addHandler(diar_handler)
        if prior_level == logging.NOTSET or prior_level > logging.INFO:
            diar_logger.setLevel(logging.INFO)

        try:
            diar_task = loop.run_in_executor(
                None, run_diarization_and_merge, process_path, segments, num_speakers
            )

            # Sentinel marks end-of-stream so we don't wait a full 15s
            # timeout after the task actually finishes.
            def _on_done(_fut):
                try:
                    loop.call_soon_threadsafe(log_queue.put_nowait, _SENTINEL)
                except RuntimeError:
                    pass

            diar_task.add_done_callback(_on_done)

            while True:
                if await request.is_disconnected():
                    # The worker thread cannot be cancelled; it will
                    # finish in the background.  The finally block below
                    # detaches our handler so its records stop arriving.
                    yield _sse("error", "Client disconnected")
                    return
                try:
                    item = await asyncio.wait_for(log_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield _sse("progress", "⏳ (diarizing...)")
                    continue
                if item is _SENTINEL:
                    break
                yield _sse("progress", item)

            # Drain anything that raced the sentinel.
            while not log_queue.empty():
                item = log_queue.get_nowait()
                if item is not _SENTINEL:
                    yield _sse("progress", item)

            try:
                segments = await diar_task
                yield _sse("progress", "✅ Diarization done.")
            except Exception as e:
                logger.exception("Diarization failed")
                yield _sse("progress", f"⚠️ Diarization failed: {e}")
                segments = [
                    {"start": s["start"], "end": s["end"], "text": s["text"]}
                    for s in segments
                ]
        finally:
            diar_logger.removeHandler(diar_handler)
            diar_logger.setLevel(prior_level)

    openai_result = transform_to_openai_verbose(full_text, segments, detected_lang)

    if response_format in ("json", "verbose_json"):
        yield _sse("result", json.dumps(openai_result))
        return

    if response_format == "text":
        if diarize and segments:
            collapse_bool = collapse.lower() in ("true", "1", "yes")
            lines = []
            for speaker, text in _group_by_speaker(segments):
                if collapse_bool:
                    text = " ".join(text.split())
                lines.append(f"{speaker}:\n{text}")
            yield _sse("result", "\n\n".join(lines))
            return
        else:
            yield _sse("result", openai_result["text"])
            return

    yield _sse("error", f"Unsupported response_format: {response_format}")


async def _stream_server(
    request, process_path, language, response_format, temperature, collapse,
):
    """whisper-server POST with heartbeat SSE while waiting."""
    with open(process_path, "rb") as f:
        files = {"file": (os.path.basename(process_path), f, "audio/mpeg")}
        data = {"response-format": "json"}
        if temperature is not None:
            data["temperature"] = str(temperature)
        if language is not None:
            data["language"] = language

        async with httpx.AsyncClient(timeout=600.0) as client:
            send_task = asyncio.create_task(
                client.post(WHISPER_SERVER_URL, files=files, data=data)
            )
            try:
                while True:
                    if await request.is_disconnected():
                        send_task.cancel()
                        yield _sse("error", "Client disconnected")
                        return
                    try:
                        resp = await asyncio.wait_for(asyncio.shield(send_task), timeout=15.0)
                        break
                    except asyncio.TimeoutError:
                        yield _sse("progress", "⏳ (transcribing...)")
                        continue
            except asyncio.CancelledError:
                send_task.cancel()
                raise
            resp = await send_task

    if resp.status_code != 200:
        yield _sse("error", f"Whisper server returned {resp.status_code}: {resp.text[:200]}")
        return

    yield _sse("progress", "✅ Transcription done.")

    try:
        server_data = resp.json()
    except Exception:
        server_data = {"text": resp.text.strip()}

    if response_format == "json":
        yield _sse("result", json.dumps(server_data))
        return
    else:
        transcription = server_data.get("text", resp.text.strip())
        if collapse.lower() in ("true", "1", "yes"):
            transcription = " ".join(transcription.split())
        yield _sse("result", transcription)
        return


@app.post("/v1/audio/translations")
async def translate_not_implemented():
    """
    Placeholder for the translation endpoint (not implemented).

    Raises:
        HTTPException: Always returns a 501 Not Implemented error.
    """
    raise HTTPException(status_code=501, detail="Translation endpoint is not implemented.")
