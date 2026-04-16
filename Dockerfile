# 1. Base Image: Raw NVIDIA CUDA (No pre-installed Python to clash with!)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:${PYTHONPATH:-}"
ENV DEBIAN_FRONTEND=noninteractive
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 3. System Dependencies for Geospatial libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# 5. Tell UV to fetch Python 3.12 and create an isolated environment
RUN uv python install 3.12
RUN uv venv /opt/venv --python 3.12

# 6. Activate the virtual environment permanently for the container
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# 7. Copy project files into the container
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# 8. Install Python Dependencies

RUN uv pip install -e ".[sentinel2,samroad,unet,utilities,benchmarking]"