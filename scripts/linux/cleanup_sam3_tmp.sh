#!/bin/bash

############################################
# SAM3 CPU Cleanup Script
#
# Cleans up temporary files created by SAM3 wrapper
# in /tmp/sam3-cpu or /temp/sam3-cpu directories
############################################

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default directories to clean
TMP_DIRS=("/tmp/sam3-cpu" "/temp/sam3-cpu")

print_header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

get_directory_size() {
    local dir="$1"
    if [ -d "$dir" ]; then
        du -sh "$dir" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

count_files() {
    local dir="$1"
    if [ -d "$dir" ]; then
        find "$dir" -type f | wc -l
    else
        echo "0"
    fi
}

cleanup_directory() {
    local dir="$1"
    local force="$2"
    
    if [ ! -d "$dir" ]; then
        print_warning "Directory does not exist: $dir"
        return 1
    fi
    
    # Get stats before deletion
    local size=$(get_directory_size "$dir")
    local file_count=$(count_files "$dir")
    
    echo ""
    echo "Directory: $dir"
    echo "Size     : $size"
    echo "Files    : $file_count"
    
    # Confirm deletion unless force flag is set
    if [ "$force" != "true" ]; then
        echo ""
        read -p "Delete this directory? [y/N]: " -n 1 -r
        echo ""
        
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipped: $dir"
            return 1
        fi
    fi
    
    # Delete directory
    if rm -rf "$dir" 2>/dev/null; then
        print_success "Deleted: $dir (freed $size)"
        return 0
    else
        print_error "Failed to delete: $dir"
        return 1
    fi
}

cleanup_specific_video() {
    local video_name="$1"
    local force="$2"
    
    echo ""
    print_header "Cleaning up workspace for: $video_name"
    
    local found=false
    
    for base_dir in "${TMP_DIRS[@]}"; do
        local target_dir="$base_dir/$video_name"
        
        if [ -d "$target_dir" ]; then
            found=true
            cleanup_directory "$target_dir" "$force"
        fi
    done
    
    if [ "$found" = false ]; then
        print_warning "No workspace found for: $video_name"
    fi
}

cleanup_all() {
    local force="$1"
    
    print_header "SAM3 CPU Workspace Cleanup"
    
    local total_freed=0
    local total_deleted=0
    
    for dir in "${TMP_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            echo ""
            echo "Checking: $dir"
            
            # List subdirectories (video workspaces)
            local workspaces=($(find "$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null))
            
            if [ ${#workspaces[@]} -eq 0 ]; then
                print_warning "No workspaces found in $dir"
                continue
            fi
            
            echo "Found ${#workspaces[@]} workspace(s):"
            for workspace in "${workspaces[@]}"; do
                local name=$(basename "$workspace")
                local size=$(get_directory_size "$workspace")
                echo "  • $name ($size)"
            done
            
            # Confirm deletion
            if [ "$force" != "true" ]; then
                echo ""
                read -p "Delete all workspaces in $dir? [y/N]: " -n 1 -r
                echo ""
                
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_warning "Skipped: $dir"
                    continue
                fi
            fi
            
            # Delete entire base directory
            if rm -rf "$dir" 2>/dev/null; then
                print_success "Deleted: $dir"
                ((total_deleted++))
            else
                print_error "Failed to delete: $dir"
            fi
        else
            print_warning "Directory does not exist: $dir"
        fi
    done
    
    echo ""
    print_header "Cleanup Summary"
    echo "Directories deleted: $total_deleted"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] [VIDEO_NAME]

Clean up SAM3 CPU temporary workspaces.

Options:
  -a, --all           Clean all workspaces in /tmp/sam3-cpu and /temp/sam3-cpu
  -f, --force         Force deletion without confirmation
  -l, --list          List all workspaces without deleting
  -h, --help          Show this help message

Arguments:
  VIDEO_NAME          Name of specific video workspace to clean

Examples:
  # Clean specific video workspace (with confirmation)
  $0 my_video

  # Clean all workspaces (with confirmation)
  $0 --all

  # Clean all workspaces without confirmation
  $0 --all --force

  # List all workspaces
  $0 --list

EOF
}

list_workspaces() {
    print_header "SAM3 CPU Workspaces"
    
    local total_size=0
    local total_count=0
    
    for base_dir in "${TMP_DIRS[@]}"; do
        if [ -d "$base_dir" ]; then
            echo ""
            echo "Base: $base_dir"
            echo "$(get_directory_size "$base_dir") total"
            echo ""
            
            # List subdirectories
            while IFS= read -r -d '' workspace; do
                local name=$(basename "$workspace")
                local size=$(get_directory_size "$workspace")
                local files=$(count_files "$workspace")
                echo "  • $name"
                echo "    Size : $size"
                echo "    Files: $files"
                echo ""
                ((total_count++))
            done < <(find "$base_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
        fi
    done
    
    if [ $total_count -eq 0 ]; then
        print_warning "No workspaces found"
    else
        echo "Total workspaces: $total_count"
    fi
}

############################################
# Main
############################################

CLEAN_ALL=false
FORCE=false
LIST_ONLY=false
VIDEO_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            CLEAN_ALL=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -l|--list)
            LIST_ONLY=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            VIDEO_NAME="$1"
            shift
            ;;
    esac
done

# Execute based on options
if [ "$LIST_ONLY" = true ]; then
    list_workspaces
elif [ "$CLEAN_ALL" = true ]; then
    cleanup_all "$FORCE"
elif [ -n "$VIDEO_NAME" ]; then
    cleanup_specific_video "$VIDEO_NAME" "$FORCE"
else
    show_usage
    exit 1
fi

print_success "Done"
echo ""
