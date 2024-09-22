#!/bin/bash

# Paths
CURRENT_DIR="$(pwd)"
INSTALL_DIR="$(ros2 pkg prefix image_matching)/lib/python3.10/site-packages/feature_matcher"
MODELS_DIR="$CURRENT_DIR/src/feature_matcher/models/"

# Remove models directory at install/
rm -rf "$INSTALL_DIR/models/"

# Create symbolic link to models in source directory
ln -s "$MODELS_DIR" "$INSTALL_DIR/"

echo "Linked models folder"
