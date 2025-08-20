#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env
if [ -f ".env" ]; then
  echo "🔹 Loading .env file..."
  source .env
fi

: "${GOOGLE_CLOUD_PROJECT:?Need to set GOOGLE_CLOUD_PROJECT}"
: "${GOOGLE_CLOUD_REGION:?Need to set GOOGLE_CLOUD_REGION}"
: "${ARTIFACT_REGISTRY_REPO:?Need to set ARTIFACT_REGISTRY_REPO}"
: "${ARTIFACT_REGISTRY_IMAGE_NAME:?Need to set ARTIFACT_REGISTRY_IMAGE_NAME}"

GCP_PROJECT=$GOOGLE_CLOUD_PROJECT
REGION=$GOOGLE_CLOUD_REGION
REPO=$ARTIFACT_REGISTRY_REPO
IMAGE_NAME=$ARTIFACT_REGISTRY_IMAGE_NAME

# Check for image tag argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <image-tag>"
  exit 1
fi

IMAGE_TAG="$1"

# Construct full Artifact Registry image path
FULL_IMAGE_PATH="${REGION}-docker.pkg.dev/${GCP_PROJECT}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🔹 Configuring gcloud authentication for Podman..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q

echo "🔹 Building container image with Podman..."
podman build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "🔹 Tagging image for Artifact Registry..."
podman tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE_PATH}"

echo "🔹 Pushing image to Artifact Registry..."
podman push "${FULL_IMAGE_PATH}"

echo "✅ Done! Image pushed to: ${FULL_IMAGE_PATH}"
