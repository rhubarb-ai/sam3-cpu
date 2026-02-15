#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file> [overlap_frames]"
    exit 1
fi

VIDEO="$1"
OVERLAP=${2:-0}

########################################
# 1️⃣ Extract Video Metadata
########################################

WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$VIDEO")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$VIDEO")
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$VIDEO")
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")

FPS_FLOAT=$(awk "BEGIN { split(\"$FPS\", a, \"/\"); print a[1]/a[2] }")
TOTAL_FRAMES=$(awk "BEGIN { print int($FPS_FLOAT * $DURATION) }")

########################################
# 2️⃣ Memory Per Frame (OpenCV default)
########################################

BYTES_PER_FRAME=$(awk "BEGIN { print $WIDTH * $HEIGHT * 3 }")

########################################
# 3️⃣ Available RAM (Linux)
########################################

AVAILABLE_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAILABLE_BYTES=$(awk "BEGIN { print $AVAILABLE_KB * 1024 }")

# Use only X% of available RAM
RAM_USAGE_PERCENT=0.25
USABLE_BYTES=$(awk "BEGIN { print int($AVAILABLE_BYTES * $RAM_USAGE_PERCENT) }")

MAX_FRAMES_PER_CHUNK=$(awk "BEGIN { print int($USABLE_BYTES / $BYTES_PER_FRAME) }")

if [ "$MAX_FRAMES_PER_CHUNK" -lt 1 ]; then
    echo "ERROR: Not enough RAM for even one frame."
    exit 1
fi

########################################
# 4️⃣ Compute Chunking with Overlap
########################################

if [ "$OVERLAP" -ge "$MAX_FRAMES_PER_CHUNK" ]; then
    echo "ERROR: Overlap ($OVERLAP) must be smaller than max frames per chunk ($MAX_FRAMES_PER_CHUNK)"
    exit 1
fi

CHUNKS=()

STRIDE=$((MAX_FRAMES_PER_CHUNK - OVERLAP))

for ((START=0, INDEX=0; START<TOTAL_FRAMES; START+=STRIDE, INDEX++)); do

    END=$((START + MAX_FRAMES_PER_CHUNK - 1))

    if [ "$END" -ge "$TOTAL_FRAMES" ]; then
        END=$((TOTAL_FRAMES - 1))
    fi

    CHUNKS+=("$INDEX:$START:$END")

done

NUM_CHUNKS=${#CHUNKS[@]}

########################################
# 5️⃣ Summary
########################################

echo "========================================"
echo "Video: $VIDEO"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "FPS: $FPS_FLOAT"
echo "Total Frames: $TOTAL_FRAMES"
echo "Overlap Frames: $OVERLAP"
echo "----------------------------------------"
echo "Max frames per chunk (50% RAM): $MAX_FRAMES_PER_CHUNK"
echo "Total chunks: $NUM_CHUNKS"
echo "========================================"

echo ""
echo "Chunk Frame Ranges:"
for entry in "${CHUNKS[@]}"; do
    IFS=":" read IDX S E <<< "$entry"
    echo "Chunk $IDX → frames [$S - $E]"
done

########################################
# 6️⃣ Ask User What To Do
########################################

echo ""
echo "Choose action:"
echo "1) Segment video using ffmpeg"
echo "2) Save metadata to JSON"
echo "3) Do nothing"
read -p "Enter choice [1-3]: " CHOICE

########################################
# 7️⃣ Build JSON Metadata
########################################

JSON_FILE="video_chunks_metadata.json"

echo "{" > $JSON_FILE
echo "  \"video\": \"$VIDEO\"," >> $JSON_FILE
echo "  \"width\": $WIDTH," >> $JSON_FILE
echo "  \"height\": $HEIGHT," >> $JSON_FILE
echo "  \"fps\": $FPS_FLOAT," >> $JSON_FILE
echo "  \"total_frames\": $TOTAL_FRAMES," >> $JSON_FILE
echo "  \"overlap\": $OVERLAP," >> $JSON_FILE
echo "  \"chunks\": [" >> $JSON_FILE

for i in "${!CHUNKS[@]}"; do
    IFS=":" read IDX S E <<< "${CHUNKS[$i]}"
    echo "    {\"chunk_index\": $IDX, \"start_frame\": $S, \"end_frame\": $E}" >> $JSON_FILE
    if [ "$i" -lt "$((NUM_CHUNKS - 1))" ]; then
        echo "    ," >> $JSON_FILE
    fi
done

echo "  ]" >> $JSON_FILE
echo "}" >> $JSON_FILE

########################################
# 8️⃣ Option Handling
########################################

if [ "$CHOICE" == "1" ]; then

    echo "Segmenting video..."

    mkdir -p video_segments

    for entry in "${CHUNKS[@]}"; do
        IFS=":" read IDX S E <<< "$entry"

        START_TIME=$(awk "BEGIN { print $S / $FPS_FLOAT }")
        END_TIME=$(awk "BEGIN { print ($E + 1) / $FPS_FLOAT }")
        DURATION_SEG=$(awk "BEGIN { print $END_TIME - $START_TIME }")

        ffmpeg -y -i "$VIDEO" -ss $START_TIME -t $DURATION_SEG \
            -c copy "video_segments/chunk_${IDX}.mp4"
    done

    echo "Segments saved in ./video_segments"
    echo "Metadata saved to $JSON_FILE"

elif [ "$CHOICE" == "2" ]; then

    echo "Metadata saved to $JSON_FILE"

else
    echo "No action performed."

fi