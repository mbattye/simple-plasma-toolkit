#!/usr/bin/env bash
# deploy_cloudrun.sh — Build, push, and deploy the heatstl image to Cloud Run.
#
# Usage:
#   ./scripts/deploy_cloudrun.sh [IMAGE_TAG]
#
# Automated alternative: push to main → GitHub Actions workflow
#   .github/workflows/deploy-cloudrun.yml (requires GCP_PROJECT_ID + GCP_SA_KEY secrets)
#
# Prerequisites:
#   - Docker Desktop running
#   - gcloud authenticated:  gcloud auth login && gcloud auth configure-docker
#   - Artifact Registry repo created (once):
#       gcloud artifacts repositories create heatstl \
#           --repository-format=docker --location=europe-west2
#   - GCS bucket created (once):
#       gsutil mb -l europe-west2 gs://<your-bucket>
#   - Service account with Storage Object Creator on the bucket (assigned to
#     the Cloud Run service)
#
# Env vars (or edit defaults below):
#   PROJECT_ID            GCP project id
#   REGION                Cloud Run region (default europe-west2)
#   REPO                  Artifact Registry repo (default heatstl)
#   SERVICE               Cloud Run service name (default heatstl-api)
#   HEATSTL_GCS_BUCKET    Required at deploy time: bucket that stores
#                         per-request artefacts (gs://… URIs returned to Analog)
#   ENGINE_SECRET         Optional shared token (X-Engine-Token)
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-europe-west2}"
REPO="${REPO:-heatstl}"
SERVICE="${SERVICE:-heatstl-api}"
TAG="${1:-latest}"
BUCKET="${HEATSTL_GCS_BUCKET:-}"

if [[ -z "${PROJECT_ID}" ]]; then
    echo "PROJECT_ID is empty; set with: gcloud config set project <id>" >&2
    exit 1
fi
if [[ -z "${BUCKET}" ]]; then
    echo "HEATSTL_GCS_BUCKET is unset; the engine will return file:// URIs that Analog can't reach." >&2
    echo "Export HEATSTL_GCS_BUCKET=<your-bucket> before re-running, or proceed and patch later." >&2
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/heatstl-api:${TAG}"

echo "========================================"
echo "  heatstl — Cloud Run Deploy"
echo "========================================"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Image:    ${IMAGE}"
echo "  Service:  ${SERVICE}"
echo "  Bucket:   ${BUCKET:-<unset>}"
echo "========================================"

echo "[1/3] Building Docker image…"
docker build --platform linux/amd64 -t "${IMAGE}" .

echo "[2/3] Pushing to Artifact Registry…"
docker push "${IMAGE}"

echo "[3/3] Deploying to Cloud Run…"
ENV_VARS="HEATSTL_ARTIFACT_STORE=gcs"
if [[ -n "${BUCKET}" ]]; then
    ENV_VARS+=",HEATSTL_GCS_BUCKET=${BUCKET}"
fi
if [[ -n "${ENGINE_SECRET:-}" ]]; then
    ENV_VARS+=",ENGINE_SECRET=${ENGINE_SECRET}"
fi

gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --platform managed \
    --port 8080 \
    --cpu 2 \
    --memory 2Gi \
    --concurrency 4 \
    --min-instances 0 \
    --max-instances 5 \
    --timeout 900 \
    --set-env-vars "${ENV_VARS}" \
    --allow-unauthenticated

SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" \
    --format "value(status.url)" 2>/dev/null || echo "")

echo ""
echo "========================================"
echo "  Deploy complete."
if [[ -n "${SERVICE_URL}" ]]; then
    echo "  URL:      ${SERVICE_URL}"
    echo "  Health:   ${SERVICE_URL}/health"
    echo "  Docs:     ${SERVICE_URL}/docs"
    echo "  Presets:  ${SERVICE_URL}/presets"
fi
echo "========================================"
