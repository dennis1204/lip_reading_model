#!/bin/bash

# Check if the user has provided enough arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_video_path> <output_video_path>"
    exit 1
fi

# Assign command line arguments to variables
INPUT_VIDEO_PATH=$1
OUTPUT_VIDEO_PATH=$2

# Run the Python script with the provided arguments
python extractMouthOnly.py "$INPUT_VIDEO_PATH" "$OUTPUT_VIDEO_PATH"
