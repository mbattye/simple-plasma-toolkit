#!/usr/bin/env bash
# deploy_cloudrun.sh — Build, push, and deploy the heatstl image to Cloud Run.
#
# Artefact storage is Cloudflare R2 (S3-compatible), matching the convention
# used by diagnostic-designer so the-grid's asset proxy can resolve URIs from
# either engine via a single code path.
#
# Usage:
#   ./scripts/deploy_cloudrun.sh [IMAGE_TAG]
#
# Automated alternative: push to main → GitHub Actions workflow
#   .github/workflows/deploy-cloudrun.yml
#
# Prerequisites (one-time):
#   - Docker Desktop running
#   - gcloud authenticated:  gcloud auth login && gcloud auth configure-docker
#   - Artifact Registry repo created (once):
#       gcloud artifacts repositories create heatstl \
#           --repository-format=docker --location=europe-west2
#   - Cloudflare R2 bucket (shared with diagnostic-designer, e.g. analog-twins)
#   - R2 API token (Account ID + Access Key + Secret) with read/write on the
#     bucket. Create at: dash.cloudflare.com → R2 → Manage R2 API Tokens.
#
# Required env vars at deploy time:
#   PROJECT_ID                    GCP project id (or `gcloud config set project`)
#   ENGINE_S3_BUCKET              R2 bucket name (e.g. analog-twins)
#   ENGINE_S3_ENDPOINT            R2 endpoint, e.g. https://<acct>.r2.cloudflarestorage.com
#   ENGINE_S3_ACCESS_KEY_ID       R2 token access key id
#   ENGINE_S3_SECRET_ACCESS_KEY   R2 token secret access key
#
# Optional env vars:
#   REGION             Cloud Run region (default europe-west2)
#   REPO               Artifact Registry repo (default heatstl)
#   SERVICE            Cloud Run service name (default heatstl-api)
#   ENGINE_SECRET      Shared token; clients must send as X-Engine-Token
#   ENGINE_S3_REGION   R2 region (default "auto")
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-europe-west2}"
REPO="${REPO:-heatstl}"
SERVICE="${SERVICE:-heatstl-api}"
TAG="${1:-latest}"
S3_REGION="${ENGINE_S3_REGION:-auto}"

if [[ -z "${PROJECT_ID}" ]]; then
    echo "PROJECT_ID is empty; set with: gcloud config set project <id>" >&2
    exit 1
fi

missing=()
for v in ENGINE_S3_BUCKET ENGINE_S3_ENDPOINT ENGINE_S3_ACCESS_KEY_ID ENGINE_S3_SECRET_ACCESS_KEY; do
    if [[ -z "${!v:-}" ]]; then
        missing+=("$v")
    fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing required env vars: ${missing[*]}" >&2
    echo "See the comment block at the top of this script for setup details." >&2
    exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/heatstl-api:${TAG}"

echo "========================================"
echo "  heatstl — Cloud Run Deploy"
echo "========================================"
echo "  Project:     ${PROJECT_ID}"
echo "  Region:      ${REGION}"
echo "  Image:       ${IMAGE}"
echo "  Service:     ${SERVICE}"
echo "  R2 bucket:   ${ENGINE_S3_BUCKET}"
echo "  R2 endpoint: ${ENGINE_S3_ENDPOINT}"
echo "========================================"

echo "[1/3] Building Docker image…"
docker build --platform linux/amd64 -t "${IMAGE}" .

echo "[2/3] Pushing to Artifact Registry…"
docker push "${IMAGE}"

echo "[3/3] Deploying to Cloud Run…"
ENV_VARS="ENGINE_ARTIFACT_STORE=s3"
ENV_VARS+=",ENGINE_S3_BUCKET=${ENGINE_S3_BUCKET}"
ENV_VARS+=",ENGINE_S3_ENDPOINT=${ENGINE_S3_ENDPOINT}"
ENV_VARS+=",ENGINE_S3_REGION=${S3_REGION}"
if [[ -n "${ENGINE_SECRET:-}" ]]; then
    ENV_VARS+=",ENGINE_SECRET=${ENGINE_SECRET}"
fi

# Sensitive values: pass as --update-secrets so they live in Secret Manager
# rather than as plain env vars on the revision.
SECRETS="ENGINE_S3_ACCESS_KEY_ID=heatstl-r2-access-key-id:latest"
SECRETS+=",ENGINE_S3_SECRET_ACCESS_KEY=heatstl-r2-secret-access-key:latest"

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
    --update-secrets "${SECRETS}" \
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
