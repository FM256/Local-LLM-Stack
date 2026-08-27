import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, validator
from sentence_transformers import CrossEncoder

# ------------------------------------------------------------
# Configuration from environment
# ------------------------------------------------------------
MODEL_NAME = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "1024"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
PRIVACY_MODE = os.environ.get("PRIVACY_MODE", "false").lower() in ("true", "1", "yes", "on")
LOG_PREVIEW_LEN = int(os.environ.get("LOG_PREVIEW_LEN", "500"))

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
logger.info(f"Loading model: {MODEL_NAME} (max_length={MAX_LENGTH})")
try:
    model = CrossEncoder(MODEL_NAME, max_length=MAX_LENGTH)
    logger.info("Model loaded successfully.")
    model_loaded = True
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    model_loaded = False


# ------------------------------------------------------------
# Pydantic request model – accepts both 'documents' and 'texts'
# ------------------------------------------------------------
class RerankRequest(BaseModel):
    query: str
    documents: Optional[List[Union[str, Dict[str, Any]]]] = None
    texts: Optional[List[Union[str, Dict[str, Any]]]] = None

    @validator("documents", "texts", always=True)
    def check_at_least_one(cls, v, values):
        if v is None and values.get("documents") is None and values.get("texts") is None:
            raise ValueError("Either 'documents' or 'texts' must be provided")
        return v

    def get_texts(self) -> List[str]:
        source = self.documents if self.documents is not None else self.texts
        if source is None:
            return []
        extracted = []
        for item in source:
            if isinstance(item, str):
                extracted.append(item)
            elif isinstance(item, dict):
                for key in ["text", "content", "page_content", "body", "description"]:
                    if key in item and isinstance(item[key], str):
                        extracted.append(item[key])
                        break
                else:
                    extracted.append(str(item))
            else:
                extracted.append(str(item))
        return extracted


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Reranker", version="2.0")


@app.get("/health")
async def health():
    return {"status": "ok" if model_loaded else "unhealthy", "model": MODEL_NAME}


@app.get("/ready")
async def ready():
    return {"ready": model_loaded}


@app.post("/rerank")
async def rerank(request: RerankRequest):
    start_time = time.time()

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    texts = request.get_texts()
    if not texts:
        raise HTTPException(status_code=400, detail="No valid text content in documents")

    # --- Log minimal metadata (safe for private mode) ---
    log_metadata = f"doc_count={len(texts)}"
    if not PRIVACY_MODE:
        # Log query preview and document previews (only at INFO level)
        logger.info(f"Query: {request.query[:200]}{'...' if len(request.query)>200 else ''}")
        for idx, t in enumerate(texts):
            preview = t[:LOG_PREVIEW_LEN] + ("..." if len(t) > LOG_PREVIEW_LEN else "")
            logger.info(f"Doc {idx} (preview): {preview}")
    else:
        # Private mode: log nothing about content
        logger.info(f"Rerank request received: {log_metadata}")

    # --- Prediction ---
    pairs = [[request.query, t] for t in texts]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.error(
            f"Prediction failed: {e}", exc_info=not PRIVACY_MODE
        )  # no stack trace in private mode
        raise HTTPException(status_code=500, detail="Prediction error")

    # --- Convert scores ---
    if isinstance(scores, np.ndarray):
        scores_list = scores.tolist()
    else:
        scores_list = [float(s) for s in scores]

    # --- Log statistics (safe for private mode) ---
    elapsed = time.time() - start_time
    if not PRIVACY_MODE:
        logger.info(f"Computed {len(scores_list)} scores in {elapsed:.3f}s")
        if scores_list:
            logger.info(
                f"Score range: min={min(scores_list):.4f}, max={max(scores_list):.4f}, mean={sum(scores_list)/len(scores_list):.4f}"
            )
            logger.info(f"First 3 scores: {scores_list[:3]}")
    else:
        logger.info(f"Rerank completed: {len(scores_list)} scores, elapsed={elapsed:.3f}s")

    return scores_list
