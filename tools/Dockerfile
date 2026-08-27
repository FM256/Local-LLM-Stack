FROM archlinux:latest

# Install required packages
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
        ffmpeg \
        python \
        python-fastapi \
        uvicorn \
        python-requests \
        python-python-multipart \
        yt-dlp \
        pacman-contrib && \
    pacman -Scc --noconfirm

# Pull the downgrade script, used to downgrade packages to specific versions using the Arch Linux Archive (ALA)
RUN curl -fL -o /usr/local/bin/downgrade https://raw.githubusercontent.com/archlinux-downgrade/downgrade/refs/heads/main/src/downgrade && \
    chmod +x /usr/local/bin/downgrade || (echo "ERROR: downgrade script download failed" && exit 1)
RUN curl -fL -o /usr/local/bin/pacignore https://raw.githubusercontent.com/archlinux-downgrade/downgrade/refs/heads/main/src/pacignore && \
    chmod +x /usr/local/bin/pacignore || (echo "ERROR: pacignore script download failed" && exit 1)

# Downgrade the curl impersonation packages to the latest working versions with yt-dlp 2026-07-04
# (if using python bindings above 0.15.x, it will silently fail due to an internal import error
# and impersonation will not work, without showing an apparent reason)
RUN downgrade --latest --ignore always --ala-only curl-impersonate=1.5.6-1 -- --noconfirm && \
    downgrade --latest --ignore always --ala-only python-curl_cffi=0.15.0-3 -- --noconfirm

# Clean up keep the image small
RUN pacman -Scc --noconfirm

# Copy app files
COPY app.py /app/app.py
COPY transcribe /usr/local/bin/transcriber
RUN chmod +x /usr/local/bin/transcriber

WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
