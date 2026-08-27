#!/usr/bin/env python3
"""
FastAPI server that exposes the transcriber script as an HTTP API with streaming progress.
Supports both YouTube URLs and local file uploads.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import traceback

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Transcriber Tool Server")


async def run_transcriber(request: Request, args: list, source_desc: str, cleanup_dir: str = None):
    """
    Launches the transcriber script with given args, streams stderr as progress,
    and returns the final transcript. If cleanup_dir is provided, it will be deleted
    after the generator finishes. Handles client disconnection by killing the subprocess.
    """
    script_path = "/usr/local/bin/transcriber"
    if not os.path.isfile(script_path):
        raise HTTPException(status_code=500, detail="Transcriber script not found")
    if not os.access(script_path, os.X_OK):
        raise HTTPException(status_code=500, detail="Transcriber script not executable")

    async def event_generator():
        proc = None
        stderr_lines = []  # collect all stderr lines
        try:
            logger.info(f"Starting subprocess for {source_desc}")
            proc = await asyncio.create_subprocess_exec(
                script_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            while True:
                if await request.is_disconnected():
                    logger.warning(
                        f"Client disconnected while transcribing {source_desc} – killing subprocess"
                    )
                    if proc and proc.returncode is None:
                        proc.terminate()
                        try:
                            await asyncio.wait_for(proc.wait(), timeout=2.0)
                        except asyncio.TimeoutError:
                            proc.kill()
                    yield f"event: error\ndata: Client disconnected\n\n"
                    return

                try:
                    line_bytes = await proc.stderr.readline()
                except Exception as e:
                    logger.error(f"Error reading stderr: {e}\n{traceback.format_exc()}")
                    yield f"event: error\ndata: Error reading process output\n\n"
                    break
                if not line_bytes:
                    break
                try:
                    line = line_bytes.decode("utf-8").strip()
                except UnicodeDecodeError:
                    line = line_bytes.decode("latin-1").strip()
                if line:
                    stderr_lines.append(line)
                    logger.debug(f"Progress: {line}")
                    yield f"event: progress\ndata: {line}\n\n"

            stdout_bytes, _ = await proc.communicate()
            exit_code = proc.returncode
            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()

            if exit_code != 0 or not stdout:
                error_lines = []
                if exit_code != 0:
                    error_lines.append(f"Script exited with code {exit_code}")
                if not stdout:
                    error_lines.append("No transcript output produced")
                if stderr_lines:
                    error_lines.append("Error details from script:")
                    error_lines.extend(stderr_lines)

                full_error = "\n".join(error_lines)
                logger.error(f"Transcription failed for {source_desc}:\n{full_error}")

                # Send error as multi‑line SSE
                yield "event: error\n"
                for line in full_error.splitlines():
                    yield f"data: {line}\n"
                yield "\n"
                return

            logger.info(f"Transcription successful for {source_desc} (length: {len(stdout)} chars)")
            yield f"event: result\ndata: {stdout}\n\n"

        except asyncio.CancelledError:
            logger.warning(f"Generator cancelled for {source_desc} – killing subprocess")
            if proc and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    proc.kill()
            raise
        finally:
            if cleanup_dir and os.path.isdir(cleanup_dir):
                try:
                    shutil.rmtree(cleanup_dir)
                    logger.debug(f"Cleaned up temporary directory: {cleanup_dir}")
                except Exception as e:
                    logger.error(f"Failed to clean up temp dir {cleanup_dir}: {e}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------- Endpoint: YouTube URL ----------
class TranscribeRequest(BaseModel):
    url: str


@app.post("/transcribe")
async def transcribe(request: Request, transcribe_req: TranscribeRequest):
    """
    Accepts a YouTube URL, runs the transcriber script, and streams progress.
    """
    logger.info(f"Received transcription request for URL: {transcribe_req.url}")
    return await run_transcriber(
        request,
        [transcribe_req.url, "--server", "http://llm-whisper:9091"],
        f"URL {transcribe_req.url}",
    )


# ---------- Endpoint: File Upload ----------
@app.post("/transcribe_file")
async def transcribe_file(request: Request, file: UploadFile = File(...)):
    """
    Accepts an uploaded audio/video file, saves it temporarily, and streams transcription progress.
    """
    logger.info(f"Received file upload: {file.filename} (content-type: {file.content_type})")

    tmpdir = tempfile.mkdtemp(prefix="transcriber_")
    try:
        file_path = os.path.join(tmpdir, file.filename)
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)

        logger.info(
            f"Saved uploaded file to {file_path} (size: {os.path.getsize(file_path)} bytes)"
        )

        return await run_transcriber(
            request,
            [file_path, "--server", "http://llm-whisper:9091"],
            f"uploaded file {file.filename}",
            cleanup_dir=tmpdir,
        )

    except Exception as e:
        if os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)
        logger.error(f"Error processing uploaded file: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
