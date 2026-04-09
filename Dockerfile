FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    apache2-utils \
    ffmpeg \
    git \
    git-lfs \
    nginx \
    rsync \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /opt/void_template

COPY runtime-requirements.txt /tmp/runtime-requirements.txt
RUN python3 -m pip install --upgrade pip wheel setuptools && \
    python3 -m pip install --no-cache-dir -r /tmp/runtime-requirements.txt && \
    rm -rf /root/.cache/pip

RUN mkdir -p /opt/sam2 && \
    cd /opt/sam2 && \
    git init && \
    git remote add origin https://github.com/facebookresearch/sam2.git && \
    git fetch --depth 1 origin aa9b8722d0585b661ded4b3dff1bd103540554ae && \
    git checkout FETCH_HEAD && \
    SAM2_BUILD_CUDA=0 python3 -m pip install --no-cache-dir --no-build-isolation . && \
    cd /opt && \
    rm -rf /opt/sam2 /root/.cache/pip

COPY . /opt/void_template

RUN chmod +x /opt/void_template/start-void.sh /opt/void_template/restart-void.sh && \
    ln -sf /opt/void_template/restart-void.sh /usr/local/bin/restart-void

WORKDIR /workspace

EXPOSE 7862 8888

CMD ["/opt/void_template/start-void.sh"]
