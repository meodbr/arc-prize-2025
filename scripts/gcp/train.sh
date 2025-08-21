#!/bin/bash

set -euo pipefail

# Variables
PROJECT_ID="tartiflette-469509"
REGION="us-central1"

# Check for config name argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <config-name>"
  exit 1
fi

CONFIG_NAME=$1
CONFIG_FILE="scripts/gcp/config/$CONFIG_NAME.json"

if [ ! -f $CONFIG_FILE ]; then
    echo File $CONFIG_FILE not found
    exit 1
fi

# Get auth token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Send request
curl -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @"$CONFIG_FILE" \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/customJobs"