# heatstl Cloud Run image
#
# Build:
#   docker build --platform linux/amd64 -t heatstl-api .
#
# Run locally:
#   docker run -p 8080:8080 \
#       -e HEATSTL_ARTIFACT_STORE=local \
#       -e HEATSTL_ARTIFACT_DIR=/tmp/heatstl-artifacts \
#       heatstl-api
#
# In Cloud Run, set HEATSTL_ARTIFACT_STORE=gcs + HEATSTL_GCS_BUCKET=<bucket>.
# The Cloud Run service account needs `Storage Object Creator` on the bucket.

FROM python:3.12-slim

WORKDIR /app

# Native libs for VTK / pyvista / gmsh / scipy.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Matplotlib + gmsh GUI bits must not try to open a display.
ENV MPLBACKEND=Agg
ENV PYTHONUNBUFFERED=1

# Copy project metadata first so the dependency install can be cached.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install heatstl with the service extra (FastAPI + uvicorn + httpx + GCS).
# uv_build is fetched as part of the PEP 517 backend bootstrap.
RUN pip install --no-cache-dir ".[service]"

EXPOSE 8080

# `--proxy-headers` keeps client IPs sane behind Cloud Run's front-end.
CMD ["uvicorn", "heatstl.service.app:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
