#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_message "$RED" "Error: $1 is required but not installed."
        exit 1
    fi
}

# Check required commands
check_command "docker"
check_command "curl"

# Function to show usage
show_usage() {
    echo "Usage: $0 [--target-arch <arch>]"
    echo "Available architectures:"
    echo "  amd64-avx2  : x86_64 with AVX2 support (default)"
    echo "  arm64-cpu   : ARM64 CPU (e.g., Apple Silicon)"
    echo "  cuda        : NVIDIA CUDA support"
    exit 1
}

# Parse command line arguments
TARGETARCH="amd64-avx2"  # default value
while [[ $# -gt 0 ]]; do
    case $1 in
        --target-arch)
            TARGETARCH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            print_message "$RED" "Unknown argument: $1"
            show_usage
            ;;
    esac
done

# Validate target architecture
case ${TARGETARCH} in
    amd64-avx2|arm64-cpu|cuda)
        print_message "$GREEN" "Building for target architecture: ${TARGETARCH}"
        ;;
    *)
        print_message "$RED" "Unsupported target architecture: ${TARGETARCH}"
        show_usage
        ;;
esac

# Create required directories
mkdir -p models prompts results

# Function to download a model
download_model() {
    local url=$1
    local output_file=$2

    if [ -f "${output_file}" ]; then
        print_message "$YELLOW" "Model ${output_file} already exists, skipping download..."
        return
    fi

    print_message "$GREEN" "Downloading ${output_file}..."
    curl -L "${url}" -o "${output_file}"
}

# Download models
print_message "$GREEN" "Downloading models..."

# Download one model to test with
download_model "https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/resolve/main/gemma-3-27b-it-Q4_K_M.gguf?download=true" "models/gemma-3-27b-it-Q4_K_M.gguf"

# Build Docker image
print_message "$GREEN" "Building Docker image for ${TARGETARCH}..."

# Build the Docker image using the existing determinism.Dockerfile
docker build \
    -f .devops/determinism.Dockerfile \
    -t llama-cpp:${TARGETARCH} \
    --build-arg TARGETARCH=${TARGETARCH} \
    .

# Set CUDA_DEVICES based on target architecture
if [ "${TARGETARCH}" = "cuda" ]; then
    CUDA_DEVICES="0"
else
    CUDA_DEVICES="none"
fi

print_message "$GREEN" "Setup completed successfully!"
print_message "$GREEN" "You can now run the model using:"
print_message "$YELLOW" "docker run --rm \\
    -v \"\$(pwd)/models:/app/models\" \\
    -v \"\$(pwd)/results:/app/results\" \\
    -v \"\$(pwd)/prompts:/app/prompts\" \\
    -e MODEL_FILE=\"/app/models/gemma-3-27b-it-Q4_K_M.gguf\" \\
    -e PROMPT_FILE=\"sample.txt\" \\
    -e SEED=\"42\" \\
    -e CUDA_DEVICES=\"${CUDA_DEVICES}\" \\
    llama-cpp:${TARGETARCH}"

# Create a sample prompt file if it doesn't exist
if [ ! -f "prompts/sample.txt" ]; then
    echo "Tell me a story about a brave knight." > prompts/sample.txt
    print_message "$GREEN" "Created sample prompt file at prompts/sample.txt"
fi
