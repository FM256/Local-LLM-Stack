FROM python:3.11-alpine

# Bootstrap uv via pip — auditable, no remote script execution.
RUN pip install --no-cache-dir uv

# Install the requests library for HTTP calls
RUN uv pip install --system --no-cache requests

# Install curl for debugging (optional)
RUN apk add --no-cache curl

WORKDIR /app

# Copy the script and the pre‑saved database
COPY setup.py .
COPY webui.db .

ENTRYPOINT ["python", "-u", "setup.py"]
