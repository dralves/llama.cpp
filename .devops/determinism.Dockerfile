
# This Dockerfile builds a CPU-only image that runs determinism-test three times

ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.4.0
# Target the CUDA build image
ARG BASE_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM $BASE_CONTAINER AS build-determinism

ARG TARGETARCH=amd64-avx2

# M2/M3 or better, choose arm8.2-a for M1
ARG GGML_CPU_ARM_ARCH=armv8.6-a
ARG GGML_CUDA_ARCHITECTURES=89

RUN apt-get update && \
    apt-get install -y build-essential git cmake libcurl4-openssl-dev

WORKDIR /app

COPY . .
ENV CUDA_DEVICES="none"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}

# Build with CPU-only configuration
# AVX2 seems to be a common CPU architecture, so we build for it by default
RUN if [ "$TARGETARCH" = "amd64-avx2" ]; then \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS_DEFAULT=OFF -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON; \
    elif [ "$TARGETARCH" = "arm64-cpu" ]; then \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS_DEFAULT=OFF -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=${GGML_CPU_ARM_ARCH}; \
    elif [ "$TARGETARCH" = "macos-metal" ]; then \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS_DEFAULT=OFF -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_METAL=ON; \
    elif [ "$TARGETARCH" = "cuda" ]; then \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS_DEFAULT=OFF -DBUILD_SHARED_LIBS=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CUBLAS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${GGML_CUDA_ARCHITECTURES}; \
    else \
    echo "Unsupported architecture"; \
    exit 1; \
    fi && \
    cmake --build build --target llama-determinism-test -j $(nproc)

RUN mkdir -p /app/bin && cp build/bin/llama-determinism-test /app/

# Also allow an environment variable for model file if you want.
ENV MODEL_FILE=""
ENV NGL="100"

RUN apt-get update && \
    apt-get install -y libgomp1 curl \
    && mkdir -p /app/results \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Create an empty /app/models if you want to mount into it
RUN mkdir -p /app/results
RUN mkdir -p /app/prompts

# Used to check if the binary has all the shared libs it needs
# ENTRYPOINT ["/bin/bash", "-c", "ldd ./llama-determinism-test"]

ENTRYPOINT ["/bin/bash", "-c", "set -e; echo '== Running determinism-test using $MODEL_FILE =='; ./llama-determinism-test -f /app/prompts/prompt1.txt --model $MODEL_FILE -o /app/results/results.txt --seed 42 -n 1000 --device $CUDA_DEVICES -ngl $NGL && echo '=== Output from results.txt ===' && cat /app/results/results.txt"]

# Run with:
# docker run --rm \
# -v "$(pwd)/models:/app/models" \
# -v "$(pwd)/results:/app/results" \
# -v "$(pwd)/prompts:/app/prompts" \
# -e MODEL_FILE="/app/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
# <image-name>:latest

# Build with:
# docker build -f .devops/determinism.Dockerfile -t determinism-cuda:latest --build-arg TARGETARCH=cuda  .