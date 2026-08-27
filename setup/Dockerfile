FROM python:3.11-alpine

# Install the requests library for HTTP calls
RUN pip install --no-cache-dir requests

# Install curl for debugging (optional)
RUN apk add --no-cache curl

WORKDIR /app

# Copy the script and the pre‑saved database
COPY setup.py .
COPY webui.db .

ENTRYPOINT ["python", "-u", "setup.py"]
