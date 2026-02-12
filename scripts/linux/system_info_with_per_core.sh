#!/usr/bin/env bash

echo "=================================================="
echo "SYSTEM INFORMATION"
echo "=================================================="

OS="$(uname)"

echo ""
echo "Platform:"
echo "System:  $OS"
echo "Machine: $(uname -m)"

# --------------------------------------------------
# RAM INFORMATION
# --------------------------------------------------
echo ""
echo "RAM Information:"

if [[ "$OS" == "Linux" ]]; then
    TOTAL_RAM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    AVAILABLE_RAM=$(grep MemAvailable /proc/meminfo | awk '{print $2}')

    TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.2f\", $TOTAL_RAM/1024/1024}")
    AVAILABLE_RAM_GB=$(awk "BEGIN {printf \"%.2f\", $AVAILABLE_RAM/1024/1024}")

    echo "Total RAM:      ${TOTAL_RAM_GB} GB"
    echo "Available RAM:  ${AVAILABLE_RAM_GB} GB"

elif [[ "$OS" == "Darwin" ]]; then
    TOTAL_RAM=$(sysctl -n hw.memsize)
    TOTAL_RAM_GB=$(awk "BEGIN {printf \"%.2f\", $TOTAL_RAM/1024/1024/1024}")

    echo "Total RAM:      ${TOTAL_RAM_GB} GB"

    VM_STAT=$(vm_stat)
    PAGE_SIZE=$(vm_stat | grep "page size of" | awk '{print $8}')
    FREE_PAGES=$(echo "$VM_STAT" | grep "Pages free" | awk '{print $3}' | tr -d '.')
    FREE_RAM=$(echo "$FREE_PAGES * $PAGE_SIZE" | bc)

    FREE_RAM_GB=$(awk "BEGIN {printf \"%.2f\", $FREE_RAM/1024/1024/1024}")

    echo "Available RAM:  ${FREE_RAM_GB} GB"
fi

# --------------------------------------------------
# CPU INFORMATION
# --------------------------------------------------
echo ""
echo "CPU Information:"

if [[ "$OS" == "Linux" ]]; then
    PHYSICAL_CORES=$(lscpu | grep "Core(s) per socket" | awk '{print $4}')
    SOCKETS=$(lscpu | grep "Socket(s)" | awk '{print $2}')
    LOGICAL_CORES=$(nproc)

    TOTAL_PHYSICAL=$((PHYSICAL_CORES * SOCKETS))

    echo "Physical Cores: $TOTAL_PHYSICAL"
    echo "Logical Cores:  $LOGICAL_CORES"

    MAX_FREQ=$(lscpu | grep "CPU max MHz" | awk '{print $4}')
    MIN_FREQ=$(lscpu | grep "CPU min MHz" | awk '{print $4}')

    if [[ ! -z "$MAX_FREQ" ]]; then
        echo "Max Frequency:  ${MAX_FREQ} MHz"
        echo "Min Frequency:  ${MIN_FREQ} MHz"
    fi

elif [[ "$OS" == "Darwin" ]]; then
    PHYSICAL_CORES=$(sysctl -n hw.physicalcpu)
    LOGICAL_CORES=$(sysctl -n hw.logicalcpu)
    MAX_FREQ=$(sysctl -n hw.cpufrequency)

    MAX_FREQ_MHZ=$(awk "BEGIN {printf \"%.2f\", $MAX_FREQ/1000000}")

    echo "Physical Cores: $PHYSICAL_CORES"
    echo "Logical Cores:  $LOGICAL_CORES"
    echo "Max Frequency:  ${MAX_FREQ_MHZ} MHz"
fi

# --------------------------------------------------
# PER-CORE FREQUENCY
# --------------------------------------------------
echo ""
echo "Per-Core Current Frequency:"

if [[ "$OS" == "Linux" ]]; then
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        core_id=$(basename "$cpu" | sed 's/cpu//')

        if [[ -f "$cpu/cpufreq/scaling_cur_freq" ]]; then
            freq=$(cat "$cpu/cpufreq/scaling_cur_freq")
            freq_mhz=$(awk "BEGIN {printf \"%.2f\", $freq/1000}")
            echo "Core $core_id: ${freq_mhz} MHz"
        else
            # fallback using /proc/cpuinfo
            freq=$(grep -m1 "cpu MHz" /proc/cpuinfo | awk '{print $4}')
            echo "Core $core_id: ${freq} MHz (fallback)"
        fi
    done
elif [[ "$OS" == "Darwin" ]]; then 
    per_core_freq=$(sysctl -n hw.cpufrequency_per_core) 
    echo "Per-Core Frequencies:" 
    for i in "${!per_core_freq[@]}"; do 
        freq_mhz=$(awk "BEGIN {printf \"%.2f\", ${per_core_freq[i]}/1000000}") 
        echo "Core $i: ${freq_mhz} MHz" 
    done 
fi

echo ""
echo "=================================================="