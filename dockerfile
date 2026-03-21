FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAX_JOBS=32

# System dependencies
RUN apt-get update && apt-get install -y \
    wget git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -L https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
    -o /usr/local/bin/micromamba && \
    chmod +x /usr/local/bin/micromamba

# Create env with python + cuda-toolkit
RUN micromamba create -n truthrl-verl python=3.10 -c conda-forge -y && \
    micromamba install -n truthrl-verl \
        -c nvidia/label/cuda-12.4.0 cuda-toolkit -y && \
    micromamba clean --all --yes

# Activate env for all subsequent RUN commands
ENV PATH="/opt/conda/envs/truthrl-verl/bin:$PATH"

# Clone repo
RUN git clone https://github.com/muyupan/TruthRL.git /workspace/my-repo

WORKDIR /workspace/my-repo

# Run the install script with MEGATRON and SGLANG flags
RUN cd training/verl && \
    USE_MEGATRON=0 USE_SGLANG=1 bash scripts/install_vllm_sglang_mcore.sh && \
    rm -f *.whl  # clean up downloaded wheel files

# Override conflicting versions with your known-working pins
RUN pip install \
    numpy==1.26.1 \
    opentelemetry-sdk==1.26.0 \
    opentelemetry-sdk==1.26.0 \
    tensordict==0.8.1 \
    click==8.2.1

# Install verl from source
RUN cd training/verl && pip install --no-deps -e .

RUN pip install \
    opentelemetry-sdk==1.39.1 \
    accelerate==1.13.0 \
    datasets==4.8.1

