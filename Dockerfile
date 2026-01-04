# 1. Base Image: Ubuntu 22.04
FROM ubuntu:22.04

# 2. Metadata
LABEL name="AQUAH"
LABEL maintainer="Songkun Yan <skyan@ou.edu>"
LABEL version="0.1"

# 3. Environment Variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install System Dependencies (Combined EF5 + Colab requirements)
# We added:
# - pandoc, texlive-xetex: From your Colab notebook (for report generation)
# - libgeos-dev, libproj-dev: REQUIRED for 'cartopy' to install correctly
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    build-essential \
    make \
    libgeotiff-dev \
    dh-autoreconf \
    autotools-dev \
    autoconf \
    automake \
    libtool \
    pkg-config \
    python3 \
    python3-pip \
    python-is-python3 \
    wget \
    vim \
    nano \
    # --- Added from your Colab ---
    pandoc \
    texlive-xetex \
    # --- Added for Cartopy/Rasterio support ---
    libgeos-dev \
    libproj-dev \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Build EF5 Hydrological Model (Same as before)
WORKDIR /EF5
RUN git clone https://github.com/HyDROSLab/EF5.git . && \
    autoreconf --force --install && \
    ./configure CXXFLAGS="-Wall -O2 -g" CFLAGS="-Wall -O2 -g" && \
    sed -i 's/-Werror//g' Makefile && \
    make -j$(nproc)

# 6. Setup Python Environment
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
# Upgrade pip first to avoid binary incompatibility
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 7. Copy Application Code
# COPY . .

# 8. Link EF5 executable
RUN ln -sf /EF5 /app/EF5

# 9. Default Command
CMD ["/bin/bash"]