#!/usr/bin/env bash
#
# setup.sh - Automated setup script for SAM3 CPU
# 
# This script installs all necessary dependencies for SAM3 CPU:
# - uv package manager
# - Python dependencies via uv
# - System dependencies (ffmpeg)
# - Project in development mode
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
print_banner() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                                                           ║${NC}"
    echo -e "${BLUE}║           SAM3 CPU - Automated Setup Script              ║${NC}"
    echo -e "${BLUE}║                                                           ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Check if running on supported OS
check_os() {
    log_info "Checking operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        log_success "Detected: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        log_success "Detected: macOS"
    else
        log_error "Unsupported operating system: $OSTYPE"
        log_error "This script supports Linux and macOS only."
        exit 1
    fi
}

# Check Python version
check_python() {
    log_info "Checking Python version..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed!"
        log_error "Please install Python 3.12 or higher and try again."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
        log_error "Python 3.12+ required, but found Python $PYTHON_VERSION"
        log_error "Please upgrade Python and try again."
        exit 1
    fi
    
    log_success "Python $PYTHON_VERSION detected"
}

# Install system dependencies
install_system_deps() {
    log_info "Checking system dependencies..."
    
    # Check ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        log_warning "ffmpeg not found. Installing..."
        
        if [ "$OS" == "linux" ]; then
            if command -v apt-get &> /dev/null; then
                log_info "Using apt-get to install ffmpeg..."
                sudo apt-get update
                sudo apt-get install -y ffmpeg
            elif command -v yum &> /dev/null; then
                log_info "Using yum to install ffmpeg..."
                sudo yum install -y ffmpeg
            elif command -v dnf &> /dev/null; then
                log_info "Using dnf to install ffmpeg..."
                sudo dnf install -y ffmpeg
            else
                log_error "Could not find package manager to install ffmpeg"
                log_error "Please install ffmpeg manually and retry"
                exit 1
            fi
        elif [ "$OS" == "macos" ]; then
            if command -v brew &> /dev/null; then
                log_info "Using Homebrew to install ffmpeg..."
                brew install ffmpeg
            else
                log_error "Homebrew not found. Please install Homebrew:"
                log_error "/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
        fi
        
        log_success "ffmpeg installed"
    else
        log_success "ffmpeg already installed"
    fi
    
    # Check ffprobe
    if ! command -v ffprobe &> /dev/null; then
        log_warning "ffprobe not found (usually comes with ffmpeg)"
        log_warning "If you encounter issues, please install ffmpeg correctly"
    else
        log_success "ffprobe found"
    fi
}

# Install uv package manager
install_uv() {
    log_info "Checking uv package manager..."
    
    if command -v uv &> /dev/null; then
        UV_VERSION=$(uv --version 2>&1 | head -n1 || echo "unknown")
        log_success "uv already installed: $UV_VERSION"
        return 0
    fi
    
    log_warning "uv not found. Installing..."
    
    # Install uv
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        log_success "uv installed successfully"
        
        # Add to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        # Verify installation
        if command -v uv &> /dev/null; then
            UV_VERSION=$(uv --version 2>&1 | head -n1)
            log_success "uv version: $UV_VERSION"
        else
            log_error "uv installation succeeded but command not found in PATH"
            log_error "Please add ~/.cargo/bin to your PATH and run this script again"
            log_error "Example: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
            exit 1
        fi
    else
        log_error "Failed to install uv"
        log_error "Please install manually: https://github.com/astral-sh/uv"
        exit 1
    fi
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    if ! command -v uv &> /dev/null; then
        log_error "uv not found in PATH"
        log_error "Please ensure uv is installed and in your PATH"
        exit 1
    fi
    
    # Install project in development mode
    log_info "Installing SAM3 CPU in development mode..."
    if uv pip install -e .; then
        log_success "Python dependencies installed"
    else
        log_error "Failed to install Python dependencies"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check if sam3 can be imported
    if python3 -c "from sam3 import Sam3; print('✓ Sam3 import successful')" 2>/dev/null; then
        log_success "SAM3 import verification passed"
    else
        log_error "Failed to import sam3 module"
        log_error "Installation may be incomplete"
        exit 1
    fi
    
    # Check PyTorch
    if python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null; then
        log_success "PyTorch verification passed"
        
        # Check CUDA availability
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_AVAILABLE" == "True" ]; then
            log_success "CUDA is available for GPU acceleration"
        else
            log_info "CUDA not available - will run on CPU"
        fi
    else
        log_warning "PyTorch not found or not working correctly"
    fi
}

# Create output directories
create_directories() {
    log_info "Creating output directories..."
    
    mkdir -p outputs
    mkdir -p temp
    
    log_success "Directories created"
}

# Print next steps
print_next_steps() {
    echo ""
    echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}║              Setup completed successfully! 🎉             ║${NC}"
    echo -e "${GREEN}║                                                           ║${NC}"
    echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo ""
    echo -e "  ${YELLOW}1.${NC} Run examples:"
    echo -e "     make run-all                    # Run all 9 scenarios"
    echo -e "     make run-example EXAMPLE=a      # Run specific example"
    echo ""
    echo -e "  ${YELLOW}2.${NC} Run tests:"
    echo -e "     make test                       # Run all tests"
    echo -e "     make test-fast                  # Run fast tests only"
    echo ""
    echo -e "  ${YELLOW}3.${NC} Get help:"
    echo -e "     make help                       # Show all commands"
    echo -e "     make info                       # Show project info"
    echo ""
    echo -e "  ${YELLOW}4.${NC} Quick start with Python:"
    echo ""
    echo -e "     ${GREEN}from sam3 import Sam3${NC}"
    echo -e "     ${GREEN}sam3 = Sam3(verbose=True)${NC}"
    echo -e "     ${GREEN}result = sam3.process_image(${NC}"
    echo -e "     ${GREEN}    image_path=\"assets/images/truck.jpg\",${NC}"
    echo -e "     ${GREEN}    prompts=[\"truck\", \"wheel\"]${NC}"
    echo -e "     ${GREEN})${NC}"
    echo ""
    echo -e "${BLUE}Documentation:${NC} See README.md for full API documentation"
    echo ""
}

# Main setup flow
main() {
    print_banner
    
    # Check prerequisites
    check_os
    check_python
    
    # Install dependencies
    install_system_deps
    install_uv
    install_python_deps
    
    # Setup project
    create_directories
    verify_installation
    
    # Done!
    print_next_steps
}

# Run main function
main
