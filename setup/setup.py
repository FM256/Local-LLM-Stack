#!/usr/bin/env python3
import json
import os
import shutil
import sys
import time
from pathlib import Path

import requests

# -------------------------------------------
# Model lists per SETUP_TYPE
# -------------------------------------------

MODEL_LISTS = {
    "cpu": [
        "qwen2.5:3b",
        "mxbai-embed-large",
        "gemma4:e4b",
        "evalengine/unbound-e4b",
    ],
    "8gb": [
        "qwen2.5:3b",
        "mxbai-embed-large",
        "gemma4:e4b",
        "gemma4:12b",
        "evalengine/unbound-e4b",
    ],
    "40gb": [
        "qwen2.5:3b",
        "mxbai-embed-large",
        "gemma4:12b",
        "gemma4:26b",
        "qwen3.6:35b",
    ],
}

# -------------------------------------------
# Environment variables (with defaults)
# -------------------------------------------
SETUP_TYPE = os.getenv("SETUP_TYPE", "8gb").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://llm-ollama:11434")
SKIP_DB_COPY = os.getenv("SKIP_DB_COPY", "false").lower() in ("true", "1", "yes", "on")

# Paths inside the container
DB_SOURCE = "/app/webui.db"
DB_DEST = "/open-webui-data/webui.db"
MARKER_FILE = "/open-webui-data/.setup_done"


# -------------------------------------------
# Helper functions
# -------------------------------------------
def wait_for_ollama(timeout=120):
    print("⏳ Waiting for Ollama to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            print(f"DEBUG: status = {resp.status_code}")
            if resp.status_code == 200:
                print("✅ Ollama is ready.")
                return True
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: exception: {e}")
            pass
        time.sleep(3)
    print("❌ Timed out waiting for Ollama.")
    return False


def model_exists(model_name):
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model_name}, timeout=10)
        return resp.status_code == 200
    except:
        return False


def pull_model(model_name):
    print(f"⬇️  Pulling model: {model_name} ...", flush=True)
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/pull", json={"name": model_name}, stream=True, timeout=600
        )
        if resp.status_code != 200:
            print(f"❌ Failed to pull {model_name}: HTTP {resp.status_code}", flush=True)
            return False

        last_print = 0.0
        PROGRESS_INTERVAL = 5.0

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
                status = data.get("status")
                if status == "success":
                    print(f"✅ Pulled {model_name} successfully.", flush=True)
                    return True
                elif status == "error":
                    error_msg = data.get("error", "unknown error")
                    print(f"❌ Error pulling {model_name}: {error_msg}", flush=True)
                    return False

                now = time.time()
                if now - last_print >= PROGRESS_INTERVAL:
                    total = data.get("total")
                    completed = data.get("completed")
                    if total and completed:
                        pct = completed / total * 100
                        print(f"  {status} {pct:.1f}% ({completed}/{total})", flush=True)
                    else:
                        print(f"  {status}", flush=True)
                    last_print = now

            except json.JSONDecodeError:
                pass

        print(f"❌ Pull of {model_name} ended without final status.", flush=True)
        return False

    except Exception as e:
        print(f"❌ Error pulling {model_name}: {e}", flush=True)
        return False


# -------------------------------------------
# Main logic
# -------------------------------------------
def main():
    # ---------- Database setup ----------
    if SKIP_DB_COPY:
        print("⏭️  SKIP_DB_COPY is set. Skipping database copy step.")
        # Create marker so we don't run again
        if not os.path.exists(MARKER_FILE):
            Path(MARKER_FILE).parent.mkdir(parents=True, exist_ok=True)
            with open(MARKER_FILE, "w") as f:
                f.write("done")
            print("✅ Marker file created (setup marked as completed).")
        else:
            print("✅ Marker file already exists.")
    else:
        # Normal mode: copy database if not already done
        if not os.path.exists(MARKER_FILE):
            if os.path.exists(DB_SOURCE):
                print("📂 Copying configuration database into Open WebUI volume...")
                Path(DB_DEST).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(DB_SOURCE, DB_DEST)
                print("✅ Database copied.")
                with open(MARKER_FILE, "w") as f:
                    f.write("done")
                print("🎉 WebUI Setup completed successfully.")
            else:
                # DB missing → warn, don't create marker, but exit cleanly
                print(
                    "⚠️  No webui.db found in setup container. Skipping copy – will retry on next run."
                )
                # Do NOT create marker – this will cause the script to run again next time.
                # Exit with success so the compose stack continues.
        else:
            print("✅ WebUI Setup already completed.")

    # ---------- Ollama setup (always runs) ----------
    if not wait_for_ollama():
        print("❌ Ollama not available. Exiting with error.")
        sys.exit(1)

    models_to_pull = MODEL_LISTS.get(SETUP_TYPE, MODEL_LISTS["8gb"])
    print(f"📦 Setup type: {SETUP_TYPE} → checking {len(models_to_pull)} models")

    for model in models_to_pull:
        if model_exists(model):
            print(f"✅ Model {model} already exists.")
        else:
            if not pull_model(model):
                print(f"⚠️  Could not pull {model}! Continuing anyway...")

    print("✅ Ollama Setup completed. Exiting.")


if __name__ == "__main__":
    main()
