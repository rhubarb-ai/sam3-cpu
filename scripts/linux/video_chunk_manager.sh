#!/usr/bin/env bash

############################################
# Usage
############################################

if [ -z "$1" ]; then
    echo "Usage: $0 <video_file> [overlap_frames]"
    exit 1
fi

VIDEO="$1"
OVERLAP=${2:-1}   # default overlap = 1

BASENAME=$(basename "$VIDEO")
NAME="${BASENAME%.*}"

# Use only X% of available RAM
RAM_USAGE_PERCENT=0.25

############################################
# Extract Video Metadata
############################################

WIDTH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$VIDEO")
HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$VIDEO")
FPS_RAW=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$VIDEO")
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO")

FPS=$(awk "BEGIN { split(\"$FPS_RAW\", a, \"/\"); print a[1]/a[2] }")
TOTAL_FRAMES=$(awk "BEGIN { print int($FPS * $DURATION) }")

############################################
# Memory Planning (OpenCV CV_8UC3)
############################################

BYTES_PER_FRAME=$(awk "BEGIN { print $WIDTH * $HEIGHT * 3 }")

AVAILABLE_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
AVAILABLE_BYTES=$(awk "BEGIN { print $AVAILABLE_KB * 1024 }")
USABLE_BYTES=$(awk "BEGIN { print int($AVAILABLE_BYTES * $RAM_USAGE_PERCENT) }")

MAX_FRAMES_RAM=$(awk "BEGIN { print int($USABLE_BYTES / $BYTES_PER_FRAME) }")

if [ "$OVERLAP" -ge "$MAX_FRAMES_RAM" ]; then
    echo "ERROR: Overlap must be smaller than RAM-safe chunk size."
    exit 1
fi

echo "============================================"
echo "Video: $VIDEO"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "FPS: $FPS"
echo "Total Frames: $TOTAL_FRAMES"
echo "RAM-safe frames per chunk: $MAX_FRAMES_RAM"
echo "Overlap: $OVERLAP"
echo "============================================"

############################################
# Directory Setup (created only if needed)
############################################

BASE_DIR="./video_segment/$NAME"
CHUNK_DIR="$BASE_DIR/chunks"
META_DIR="$BASE_DIR/metadata"

############################################
# Helper: Generate Frame Chunks
############################################

generate_chunks_by_frame_size() {
    local CHUNK_SIZE=$1
    local STRIDE=$((CHUNK_SIZE - OVERLAP))
    CHUNKS=()

    for ((START=0, IDX=0; START<TOTAL_FRAMES; START+=STRIDE, IDX++)); do
        END=$((START + CHUNK_SIZE - 1))
        if [ "$END" -ge "$TOTAL_FRAMES" ]; then
            END=$((TOTAL_FRAMES - 1))
        fi
        CHUNKS+=("$IDX:$START:$END")
    done
}

############################################
# Menu
############################################

echo ""
echo "Choose option:"
echo "1) Save video in RAM-safe chunks"
echo "2) Equal frame-length chunks"
echo "3) Equal time-length chunks"
echo "4) Save metadata only"
echo "5) Do nothing"

read -p "Enter choice [1-5]: " OPTION

############################################
# OPTION 1
############################################

if [ "$OPTION" == "1" ]; then

    generate_chunks_by_frame_size $MAX_FRAMES_RAM

############################################
# OPTION 2
############################################

elif [ "$OPTION" == "2" ]; then

    read -p "Enter number of equal chunks: " NUM
    CHUNK_SIZE=$(awk "BEGIN { print int($TOTAL_FRAMES / $NUM) }")

    if [ "$CHUNK_SIZE" -gt "$MAX_FRAMES_RAM" ]; then
        echo "ERROR: Chunk size exceeds RAM-safe limit."
        exit 1
    fi

    generate_chunks_by_frame_size $CHUNK_SIZE

############################################
# OPTION 3
############################################

elif [ "$OPTION" == "3" ]; then

    read -p "Enter chunk duration (seconds): " SEC

    FRAMES_PER_TIME=$(awk "BEGIN { print int($SEC * $FPS) }")

    if [ "$FRAMES_PER_TIME" -gt "$MAX_FRAMES_RAM" ]; then
        echo "ERROR: Time chunk exceeds RAM-safe frame limit."
        exit 1
    fi

    generate_chunks_by_frame_size $FRAMES_PER_TIME

############################################
# OPTION 4 (metadata only)
############################################

elif [ "$OPTION" == "4" ]; then

    generate_chunks_by_frame_size $MAX_FRAMES_RAM

############################################
# OPTION 5
############################################

elif [ "$OPTION" == "5" ]; then
    echo "No action performed."
    exit 0
else
    echo "Invalid option."
    exit 1
fi

############################################
# If option 1–4 → proceed
############################################

mkdir -p "$CHUNK_DIR"
mkdir -p "$META_DIR"

############################################
# Save Video Chunks (only option 1–3)
############################################

if [[ "$OPTION" == "1" || "$OPTION" == "2" || "$OPTION" == "3" ]]; then

    echo "Saving video chunks..."

    for entry in "${CHUNKS[@]}"; do
        IFS=":" read IDX S E <<< "$entry"

        START_TIME=$(awk "BEGIN { print $S / $FPS }")
        END_TIME=$(awk "BEGIN { print ($E + 1) / $FPS }")
        DUR=$(awk "BEGIN { print $END_TIME - $START_TIME }")

        ffmpeg -loglevel error -y -i "$VIDEO" \
            -ss $START_TIME -t $DUR -c copy \
            "$CHUNK_DIR/chunk_${IDX}.mp4"
    done
fi

############################################
# Metadata Save
############################################

if [[ "$OPTION" == "1" || "$OPTION" == "2" || "$OPTION" == "3" || "$OPTION" == "4" ]]; then

    read -p "Save metadata format (json/csv/tsv): " FORMAT

    META_FILE="$META_DIR/chunk_metadata.$FORMAT"

    if [ "$FORMAT" == "json" ]; then
        echo "{" > $META_FILE
        echo "  \"video\": \"$VIDEO\"," >> $META_FILE
        echo "  \"chunks\": [" >> $META_FILE
        for i in "${!CHUNKS[@]}"; do
            IFS=":" read IDX S E <<< "${CHUNKS[$i]}"
            echo "    {\"chunk\": $IDX, \"start\": $S, \"end\": $E}" >> $META_FILE
            if [ "$i" -lt "$(( ${#CHUNKS[@]} - 1 ))" ]; then
                echo "    ," >> $META_FILE
            fi
        done
        echo "  ]" >> $META_FILE
        echo "}" >> $META_FILE

    elif [ "$FORMAT" == "csv" ]; then
        echo "chunk,start_frame,end_frame" > $META_FILE
        for entry in "${CHUNKS[@]}"; do
            IFS=":" read IDX S E <<< "$entry"
            echo "$IDX,$S,$E" >> $META_FILE
        done

    elif [ "$FORMAT" == "tsv" ]; then
        echo -e "chunk\tstart_frame\tend_frame" > $META_FILE
        for entry in "${CHUNKS[@]}"; do
            IFS=":" read IDX S E <<< "$entry"
            echo -e "$IDX\t$S\t$E" >> $META_FILE
        done

    else
        echo "Unsupported format."
    fi

    echo "Metadata saved to $META_FILE"
fi

echo "Done."