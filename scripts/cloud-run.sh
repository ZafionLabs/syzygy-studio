#!/usr/bin/env bash
# Release terrain-match in separate steps:
#   ./scripts/cloud-run.sh registry
#   ./scripts/cloud-run.sh build
#   ./scripts/cloud-run.sh push
#   ./scripts/cloud-run.sh deploy
#
# The image address is gcr.io/PROJECT/terrain-match:latest. Google shut down
# Container Registry writes, so "registry" creates the Artifact Registry
# repository that still serves that hostname.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
if [[ -z "${PROJECT}" || "${PROJECT}" == "(unset)" ]]; then
  echo "Set a project first: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

REGION="${REGION:-us-central1}"
IMAGE="gcr.io/${PROJECT}/terrain-match:latest"

case "${1:-}" in
  registry)
    gcloud services enable artifactregistry.googleapis.com
    if ! gcloud artifacts repositories describe gcr.io --location=us >/dev/null 2>&1; then
      gcloud artifacts repositories create gcr.io \
        --repository-format=docker \
        --location=us \
        --description="terrain-match images"
    fi
    gcloud auth configure-docker --quiet
    echo "Registry is ready for ${IMAGE}"
    ;;
  build)
    docker build -t "${IMAGE}" .
    echo "Built ${IMAGE}"
    ;;
  push)
    docker push "${IMAGE}"
    echo "Pushed ${IMAGE}"
    ;;
  deploy)
    gcloud run deploy terrain-match \
      --image "${IMAGE}" \
      --region "${REGION}" \
      --allow-unauthenticated \
      --port 8080 \
      --memory 2Gi \
      --cpu 1 \
      --timeout 300 \
      --min-instances 1 \
      --max-instances 1
    ;;
  *)
    echo "Usage: $0 registry|build|push|deploy" >&2
    exit 1
    ;;
esac
