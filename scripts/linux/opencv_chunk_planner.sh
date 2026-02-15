#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file>"
    exit 1
fi

VIDEO="$1"

########################################
# 1️ Extract Video Metadata
########################################

WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$VIDEO")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$VIDEO")
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$VIDEO")
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")

if [ -z "$WIDTH" ] || [ -z "$HEIGHT" ]; then
    echo "Error: Could not read video metadata."
    exit 1
fi

# Convert FPS fraction (e.g., 30000/1001)
FPS_FLOAT=$(awk "BEGIN { split(\"$FPS\", a, \"/\"); print a[1]/a[2] }")

TOTAL_FRAMES=$(awk "BEGIN { print int($FPS_FLOAT * $DURATION) }")

########################################
# 2️ Memory Per Frame (OpenCV default)
########################################
# CV_8UC3 → width × height × 3 bytes

BYTES_PER_FRAME=$(awk "BEGIN { print $WIDTH * $HEIGHT * 3 }")

########################################
# 3️ Detect Available RAM
########################################
# Linux: use /proc/meminfo

AVAILABLE_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAILABLE_BYTES=$(awk "BEGIN { print $AVAILABLE_KB * 1024 }")

# Use only X% of available RAM
RAM_USAGE_PERCENT=0.25
USABLE_BYTES=$(awk "BEGIN { print int($AVAILABLE_BYTES * $RAM_USAGE_PERCENT) }")

########################################
# 4️ Compute Frames Per Chunk
########################################

MAX_FRAMES_PER_CHUNK=$(awk "BEGIN { print int($USABLE_BYTES / $BYTES_PER_FRAME) }")

if [ "$MAX_FRAMES_PER_CHUNK" -lt 1 ]; then
    echo "ERROR: Not enough RAM to hold even one frame."
    exit 1
fi

########################################
# 5️ Compute Number of Chunks
########################################

NUM_CHUNKS=$(awk "BEGIN { print int(($TOTAL_FRAMES + $MAX_FRAMES_PER_CHUNK - 1) / $MAX_FRAMES_PER_CHUNK) }")

########################################
# 6️ Print Summary
########################################

echo "========================================"
echo "Video: $VIDEO"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "FPS: $FPS_FLOAT"
echo "Duration: ${DURATION}s"
echo "Total Frames: $TOTAL_FRAMES"
echo "----------------------------------------"
echo "Memory per frame (CV_8UC3): $BYTES_PER_FRAME bytes"
echo "Available RAM: $(awk "BEGIN { printf \"%.2f\", $AVAILABLE_BYTES / (1024^3) }") GB"
echo "Using 50% RAM: $(awk "BEGIN { printf \"%.2f\", $USABLE_BYTES / (1024^3) }") GB"
echo "----------------------------------------"
echo "Max frames per chunk: $MAX_FRAMES_PER_CHUNK"
echo "Total chunks required: $NUM_CHUNKS"
echo "========================================"

########################################
# 7️ Frame Index Ranges
########################################

echo ""
echo "Suggested Frame Index Ranges:"
echo "----------------------------------------"

START=0

for ((i=0; i<NUM_CHUNKS; i++)); do
    END=$((START + MAX_FRAMES_PER_CHUNK - 1))

    if [ "$END" -ge "$TOTAL_FRAMES" ]; then
        END=$((TOTAL_FRAMES - 1))
    fi

    echo "Chunk $i → frames [$START - $END]"

    START=$((END + 1))
done