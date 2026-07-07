# Hugging Face Spaces (Docker SDK) deployment for citrus-disease-detector.
# Free CPU Spaces give 2 vCPU / 16GB RAM, so no memory workarounds are
# needed here — this runs the ORIGINAL app.py with both models enabled,
# gunicorn -w 2, and the scene-gate classifier all as originally written.

FROM python:3.11-slim

# System libraries required by opencv-python-headless and Pillow.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better Docker layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app.
COPY . .

# Hugging Face Spaces routes traffic to port 7860 by default.
ENV PORT=7860
EXPOSE 7860

# Spaces containers run as a non-root user with a restricted home dir;
# make sure the app's writable folders (SQLite DB, uploaded images) exist
# and are writable before the app starts.
RUN mkdir -p instance static/uploads && chmod -R 777 instance static/uploads

CMD ["gunicorn", "-w", "2", "--timeout", "120", "-b", "0.0.0.0:7860", "app:app"]
