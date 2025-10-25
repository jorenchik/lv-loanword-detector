#!/bin/bash

# Usage: ./package_data.sh v0_2_1

set -e  # Exit on error

VERSION=$1

if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version_tag> (e.g., v0_2_1)"
  exit 1
fi

TARGET_DIR="lv_loanword_detection_data_$VERSION"
ZIP_FILE="${TARGET_DIR}.zip"

mkdir "$TARGET_DIR"
cp -r data "$TARGET_DIR"

zip -r "$ZIP_FILE" "$TARGET_DIR"
echo "Created $ZIP_FILE"
