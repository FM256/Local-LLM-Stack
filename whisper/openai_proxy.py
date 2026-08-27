import asyncio
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import uvicorn
from diarize import diarize
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

# ----------------------------------------------------------------------
# Logging and Debug Flag
# ----------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.DEBUG)

DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
if DEBUG:
    logger.info("🔍 DEBUG mode enabled – verbose logs will be printed")

app = FastAPI()

# ----------------------------------------------------------------------
# Constants & paths
# ----------------------------------------------------------------------
WHISPER_SERVER_URL = "http://localhost:8080/inference"
WHISPER_CLI = "/app/build/bin/whisper-cli"
MODEL_PATH = "/app/models/ggml-small.bin"
FFMPEG = "/usr/bin/ffmpeg"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


# ----------------------------------------------------------------------
# Diarization helpers
# ----------------------------------------------------------------------
def find_optimal_clusters(embeddings: np.ndarray, max_speakers: int = 6) -> int:
    """
    Determines the optimal number of speaker clusters using silhouette score.

    Args:
        embeddings: A 2D numpy array of shape (n_samples, embedding_dim) containing
            the voice embeddings for each time window.
        max_speakers: The maximum number of clusters to evaluate. Defaults to 6.

    Returns:
        int: The optimal number of clusters between 2 and max_speakers (inclusive).
            Returns 2 if the number of samples is too small or clustering fails.

    Notes:
        - Silhouette score is computed for each candidate cluster count.
        - The count with the highest silhouette score is chosen.
        - If only one speaker is present (or clustering fails), the function
          gracefully returns 2 as a sensible default.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 3:
        if DEBUG:
            logger.debug(f"Too few embeddings ({n_samples}), defaulting to 2 speakers")
        return 2
    best_n = 2
    best_score = -1.0
    for n in range(2, min(max_speakers, n_samples) + 1):
        clustering = AgglomerativeClustering(n_clusters=n)
        labels = clustering.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if DEBUG:
            logger.debug(f"Silhouette score for n={n}: {score:.4f}")
        if score > best_score:
            best_score = score
            best_n = n
    if DEBUG:
        logger.debug(f"Optimal clusters: {best_n} (score {best_score:.4f})")
    return best_n


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


async def convert_to_wav(input_path: str) -> str:
    """
    Converts any audio file to a 16 kHz mono WAV using ffmpeg.

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
            cmd = [WHISPER_CLI, "-m", MODEL_PATH, "-f", process_path]

            # Add language hint if provided
            if language:
                cmd += ["-l", language]

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
                    for seg in segments:
                        speaker = seg.get("speaker", "SPEAKER_00")
                        text = seg.get("text", "")
                        if collapse_bool:
                            text = " ".join(text.split())
                        lines.append(f"{speaker}:\n{text}")
                    # Join segments with a newline between them (no extra blank lines)
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


@app.post("/v1/audio/translations")
async def translate_not_implemented():
    """
    Placeholder for the translation endpoint (not implemented).

    Raises:
        HTTPException: Always returns a 501 Not Implemented error.
    """
    raise HTTPException(status_code=501, detail="Translation endpoint is not implemented.")
