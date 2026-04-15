# 1. Base Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 2. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PATH=/usr/local/bin:$PATH
ENV DEBIAN_FRONTEND=noninteractive
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# 3. System Dependencies
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

# 5. Copy Files
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# 6. Install Python Dependencies
# Upgrade Python to 3.12
RUN /opt/conda/bin/conda install -y python=3.12
RUN uv pip install --system -e ".[sentinel2,samroad,unet,instageo,terra,utilities]"
