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

if [ ! -f "$CONFIG_FILE" ]; then
  echo "File $CONFIG_FILE not found"
  exit 1
fi

# Get auth token
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Send request and capture response
RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @"$CONFIG_FILE" \
  "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/customJobs")

echo "$RESPONSE"

# Extract the job name from JSON
# JOB_NAME=$(echo "$RESPONSE" | grep -o '"name": *"[^"]*"' | sed 's/"name": *"//;s/"//')

# Or more robustly (requires jq)
JOB_NAME=$(echo "$RESPONSE" | jq -r '.name')

if [ -z "$JOB_NAME" ]; then
  echo "❌ Failed to extract job name from response."
  exit 1
fi

echo "✅ Custom Job created: $JOB_NAME"
echo "📜 Streaming logs..."

# Stream logs using the actual job name
gcloud ai custom-jobs stream-logs "$JOB_NAME" --project="$PROJECT_ID" --region="$REGION"