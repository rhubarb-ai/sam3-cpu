#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file>"
    exit 1
fi

VIDEO="$1"

# Extract video metadata
WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$VIDEO")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$VIDEO")
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$VIDEO")
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")

# Convert FPS from fraction (e.g., 30000/1001)
FPS_FLOAT=$(awk "BEGIN { split(\"$FPS\", a, \"/\"); print a[1]/a[2] }")

# Total frames
TOTAL_FRAMES=$(awk "BEGIN { print int($FPS_FLOAT * $DURATION) }")

# OpenCV default: 3 bytes per pixel (CV_8UC3)
BYTES_PER_FRAME=$(awk "BEGIN { print $WIDTH * $HEIGHT * 3 }")

# Total bytes
TOTAL_BYTES=$(awk "BEGIN { print $BYTES_PER_FRAME * $TOTAL_FRAMES }")

# Convert to human readable
TOTAL_GB=$(awk "BEGIN { printf \"%.2f\", $TOTAL_BYTES / (1024^3) }")
TOTAL_MB=$(awk "BEGIN { printf \"%.2f\", $TOTAL_BYTES / (1024^2) }")

echo "Video: $VIDEO"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "FPS: $FPS_FLOAT"
echo "Duration: ${DURATION}s"
echo "Total Frames: $TOTAL_FRAMES"
echo "Memory per frame (OpenCV CV_8UC3): $BYTES_PER_FRAME bytes"
echo "--------------------------------------------"
echo "Total RAM required (all frames in memory):"
echo "≈ $TOTAL_MB MB"
echo "≈ $TOTAL_GB GB"