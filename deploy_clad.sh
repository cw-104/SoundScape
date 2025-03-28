#!/bin/bash

# Example usage: ./deploy_clad.sh /path/to/SoundScape-main

TARGET_DIR=$1

if [ -z "$TARGET_DIR" ]; then
  echo "Usage: $0 /path/to/SoundScape-main"
  exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
  echo " Error: Target directory $TARGET_DIR does not exist!"
  exit 1
fi

echo " Deploying CLAD integration into: $TARGET_DIR"

# Ensure backend directory exists
mkdir -p "$TARGET_DIR/src/sound_scape/backend"

# Copy backend Python modules
cp sound_scape/backend/CladModel.py "$TARGET_DIR/src/sound_scape/backend/"
cp sound_scape/backend/clad_integration.py "$TARGET_DIR/src/sound_scape/backend/"
cp sound_scape/backend/Results_CLAD_sample.py "$TARGET_DIR/src/sound_scape/backend/"

# Create /CLAD directory if not exists
mkdir -p "$TARGET_DIR/CLAD"
cp CLAD/run_clad.py "$TARGET_DIR/CLAD/"

# Optionally copy setup scripts and requirements
cp setup_clad.sh "$TARGET_DIR/"
cp requirements.txt "$TARGET_DIR/"

echo " CLAD files deployed successfully to $TARGET_DIR"

# Optional: Prompt to run venv setup
echo " You can now run: cd $TARGET_DIR && bash setup_clad.sh (if needed to create CLAD venv)"
