#!/bin/bash
set -euo pipefail

LOG_PREFIX="[VOID]"

log() {
    echo "${LOG_PREFIX} $*"
}

SOURCE_DIR=/opt/void_template
TARGET_DIR=/workspace/VOID-on-Runpod
LOG_DIR="${TARGET_DIR}/logs"
CHECKPOINT_DIR="${TARGET_DIR}/checkpoints"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export VOID_ACCESS_PORT="${VOID_ACCESS_PORT:-7862}"
export VOID_WORKSPACE_DIR="${VOID_WORKSPACE_DIR:-$TARGET_DIR}"
export VOID_BASE_MODEL_PATH="${VOID_BASE_MODEL_PATH:-$CHECKPOINT_DIR/CogVideoX-Fun-V1.5-5b-InP}"
export VOID_PASS1_PATH="${VOID_PASS1_PATH:-$CHECKPOINT_DIR/void_pass1.safetensors}"
export VOID_PASS2_PATH="${VOID_PASS2_PATH:-$CHECKPOINT_DIR/void_pass2.safetensors}"
export VOID_SAM2_CHECKPOINT="${VOID_SAM2_CHECKPOINT:-$CHECKPOINT_DIR/sam2_hiera_large.pt}"
export VOID_SAM3_HF_REPO="${VOID_SAM3_HF_REPO:-facebook/sam3}"

if [ ! -f "${TARGET_DIR}/app.py" ]; then
    log "Restoring application files to ${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"
    rsync -a "${SOURCE_DIR}/" "${TARGET_DIR}/"
else
    log "Application files already present in workspace"
fi

mkdir -p \
    "${CHECKPOINT_DIR}" \
    "${TARGET_DIR}/jobs" \
    "${LOG_DIR}" \
    "${HF_HOME}"

python_module_available() {
    local module_name="$1"

    MODULE_NAME="${module_name}" python3 - <<'PY'
import importlib
import os
import sys

try:
    importlib.import_module(os.environ["MODULE_NAME"])
except Exception:
    sys.exit(1)
sys.exit(0)
PY
}

has_hf_auth() {
    [ -n "${HF_TOKEN:-}" ] || [ -f "${HF_HOME}/token" ] || [ -f "${HOME:-/root}/.cache/huggingface/token" ]
}

install_runtime_python_packages() {
    if python_module_available sam2; then
        log "segment-anything-2 already importable"
    else
        log "Installing runtime git-based Python package: segment-anything-2"
        python3 -m pip install --no-cache-dir --no-build-isolation git+https://github.com/facebookresearch/segment-anything-2.git
    fi

    if python_module_available sam3; then
        log "sam3 already importable"
    else
        log "Installing runtime git-based Python package: sam3"
        python3 -m pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git
    fi
}

warm_sam3_hf_cache() {
    if ! python_module_available sam3; then
        return 0
    fi

    log "SAM3 package is installed. Official SAM3 weights are gated on Hugging Face (${VOID_SAM3_HF_REPO})."
    if ! has_hf_auth; then
        log "No Hugging Face auth detected. SAM3 quadmask generation will require an approved HF_TOKEN or prior 'hf auth login'."
        return 0
    fi

    log "Warming SAM3 Hugging Face cache from ${VOID_SAM3_HF_REPO}"
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["VOID_SAM3_HF_REPO"]

try:
    snapshot_download(
        repo_id=repo_id,
        token=os.environ.get("HF_TOKEN") or None,
        resume_download=True,
    )
    print(f"[VOID] SAM3 cache ready for {repo_id}")
except Exception as exc:
    print(f"[VOID] Warning: could not warm SAM3 cache from {repo_id}: {exc}")
PY
}

download_snapshot() {
    local repo_id="$1"
    local target_dir="$2"
    local label="$3"

    if [ -d "${target_dir}" ] && [ -n "$(find "${target_dir}" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
        log "${label} already cached at ${target_dir}"
        return 0
    fi

    log "Downloading ${label} from ${repo_id}"
    mkdir -p "${target_dir}"
    python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["DOWNLOAD_REPO_ID"]
target_dir = os.environ["DOWNLOAD_TARGET_DIR"]
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    token=token,
    resume_download=True,
)
PY
}

download_hf_file() {
    local repo_id="$1"
    local filename="$2"
    local local_dir="$3"
    local label="$4"

    if [ -f "${local_dir}/${filename}" ]; then
        log "${label} already cached at ${local_dir}/${filename}"
        return 0
    fi

    log "Downloading ${label} from ${repo_id}/${filename}"
    mkdir -p "${local_dir}"
    python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ["DOWNLOAD_REPO_ID"]
filename = os.environ["DOWNLOAD_FILENAME"]
local_dir = os.environ["DOWNLOAD_LOCAL_DIR"]
token = os.environ.get("HF_TOKEN") or None

hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    token=token,
    local_dir_use_symlinks=False,
    resume_download=True,
)
PY
}

download_url() {
    local url="$1"
    local output_path="$2"
    local label="$3"

    if [ -f "${output_path}" ]; then
        log "${label} already cached at ${output_path}"
        return 0
    fi

    log "Downloading ${label}"
    python3 - <<'PY'
import os
import requests

url = os.environ["DOWNLOAD_URL"]
output_path = os.environ["DOWNLOAD_OUTPUT_PATH"]

with requests.get(url, stream=True, timeout=600) as response:
    response.raise_for_status()
    with open(output_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
PY
}

export DOWNLOAD_REPO_ID="alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"
export DOWNLOAD_TARGET_DIR="${VOID_BASE_MODEL_PATH}"
download_snapshot "${DOWNLOAD_REPO_ID}" "${DOWNLOAD_TARGET_DIR}" "CogVideoX-Fun base model"

export DOWNLOAD_REPO_ID="netflix/void-model"
export DOWNLOAD_FILENAME="void_pass1.safetensors"
export DOWNLOAD_LOCAL_DIR="${CHECKPOINT_DIR}"
download_hf_file "${DOWNLOAD_REPO_ID}" "${DOWNLOAD_FILENAME}" "${DOWNLOAD_LOCAL_DIR}" "VOID Pass 1 checkpoint"

export DOWNLOAD_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
export DOWNLOAD_OUTPUT_PATH="${VOID_SAM2_CHECKPOINT}"
download_url "${DOWNLOAD_URL}" "${DOWNLOAD_OUTPUT_PATH}" "SAM2 checkpoint"

install_runtime_python_packages
warm_sam3_hf_cache

USERNAME="${VOID_USERNAME:-admin}"
PASSWORD="${VOID_PASSWORD:-void}"
TARGET_PORT="${GRADIO_SERVER_PORT}"
PROXY_PORT="${VOID_ACCESS_PORT}"

htpasswd -cb /etc/nginx/.htpasswd "${USERNAME}" "${PASSWORD}"

cat > /etc/nginx/conf.d/void-auth.conf <<EOF_CONF
server {
    listen ${PROXY_PORT};

    location / {
        auth_basic "VOID Access";
        auth_basic_user_file /etc/nginx/.htpasswd;

        proxy_pass http://127.0.0.1:${TARGET_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF_CONF

if ! grep -q "include /etc/nginx/conf.d/.*.conf;" /etc/nginx/nginx.conf; then
    sed -i '/http {/a \    include /etc/nginx/conf.d/*.conf;' /etc/nginx/nginx.conf
fi

nginx -t >/dev/null
if ! nginx -s reload >/dev/null 2>&1; then
    nginx -s stop >/dev/null 2>&1 || true
    nginx
fi

APP_LOG="${LOG_DIR}/void.log"
log "Launching VOID Gradio app on port ${TARGET_PORT}"
cd "${TARGET_DIR}"
nohup python3 app.py > "${APP_LOG}" 2>&1 &

log "VOID started. Logs: ${APP_LOG}"
log "Auth credentials -> user: ${USERNAME} password: ${PASSWORD}"
log "External access via port ${PROXY_PORT}"

if [ "${VOID_RESTART_ONLY:-0}" = "1" ]; then
    exit 0
fi

if [ -f "/start.sh" ]; then
    exec /start.sh
else
    exec tail -f "${APP_LOG}"
fi
