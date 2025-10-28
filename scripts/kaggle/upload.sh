#!/usr/bin/env bash
#
# upload.sh
# Creates a new version of a Kaggle dataset containing source python code

set -e  # Exit immediately if a command exits with a non-zero status

# === CONFIGURATION ===
DATASET_PATH="./scripts/kaggle/upload_dataset"
VERSION_MESSAGE="Updated dataset with new files"
DIR_MODE="zip"  # options: skip, zip, tar
DELETE_OLD=false

# === FILES & DIRECTORIES TO INCLUDE ===
FILES_AND_DIRS=(
    "./src"
    "./README.md"
    "./envs"
    "./pyproject.toml"
    "./LICENSE"
)

# === DATASET METADATA ===
# Must include 'dataset-metadata.json' inside the upload folder.
METADATA_FILE="./scripts/kaggle/dataset-metadata.json"



# === SCRIPT START ===

echo "🔄 Preparing upload directory: $DATASET_PATH"
rm -rf "$DATASET_PATH"
mkdir -p "$DATASET_PATH"

echo "📦 Copying files..."
for item in "${FILES_AND_DIRS[@]}"; do
  if [ -e "$item" ]; then
    echo "  - Adding $(basename "$item")"
    cp -r "$item" "$DATASET_PATH/"
  else
    echo "  ⚠️ Skipping missing: $item"
  fi
done

# Copy metadata file
if [ -f "$METADATA_FILE" ]; then
  cp "$METADATA_FILE" "$DATASET_PATH/"
else
  echo "❌ ERROR: Metadata file not found at $METADATA_FILE"
  exit 1
fi

# === UPLOAD TO KAGGLE ===
echo "🚀 Uploading new version to Kaggle..."

kaggle datasets version \
  --path "$DATASET_PATH" \
  --message "$VERSION_MESSAGE" \
  --dir-mode "$DIR_MODE" \
  ${DELETE_OLD:+--delete-old-versions}

echo "✅ Upload complete!"
