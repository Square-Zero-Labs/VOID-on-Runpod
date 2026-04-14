# VOID on Runpod

This repository wraps the upstream [`VOID`](./VOID) subtree in a Runpod-friendly Gradio app and Docker image.

The app supports:

- browser-based point selection across arbitrary frames
- Gemini-backed quadmask generation
- staged Pass 1 and Pass 2 execution
- reopening an existing saved job by name without reuploading the source video
- uploading an existing Pass 1 result and continuing directly to Pass 2
- first-boot download of the required models and SAM 2.1 checkpoint
- lazy download of the Pass 2 checkpoint only when Pass 2 is requested
- nginx basic auth in front of Gradio

## Differences From Upstream VOID

This wrapper stays close to the researchers' pipeline, but it intentionally changes two pieces of mask generation:

- `SAM 2.1` only for Stage 1 primary-object segmentation. This repo pins `facebookresearch/sam2` to a SAM 2.1-compatible commit and uses the `sam2.1_hiera_large.pt` checkpoint. Older `sam2_hiera_*.pt` checkpoints are intentionally not supported.
- Connected affected-region handling in Stage 3a. When Gemini's analysis only identifies affected areas on a few frames, this repo carries those regions across the frames in between so the grey mask stays more continuous and less patchy over time.
- Automatic VLM trajectory use in Stage 3a. When Gemini marks an affected object as moving and provides a `trajectory_path`, this repo now turns that trajectory directly into grey-mask coverage without requiring a human to draw Stage 3b trajectory input.
- A small SAM3 runtime wrapper for stability. This repo uses a patched SAM3 processor to avoid dtype and fused-kernel issues that otherwise made grey-mask generation unreliable in our environment.
- Pass 2 warped-noise temporal alignment. This repo resizes warped noise to the actual loaded clip length in latent space instead of assuming the fixed temporal window size, which avoids tensor-shape mismatches on longer padded clips.

## Build

```bash
docker build --platform=linux/amd64 -t void-runpod:latest .
```

## Setup

- Get a Gemini API Key with billing turned on - https://aistudio.google.com/api-keys
- Get approval to access the SAM3 model weights on Hugging Face - https://huggingface.co/facebook/sam3
- Get an access token with read access on Hugging Face - https://huggingface.co/settings/tokens

## Run Locally

```bash
docker run --gpus all -p 7862:7862 \
  -e VOID_USERNAME=admin \
  -e VOID_PASSWORD=void \
  -e GEMINI_API_KEY=your_gemini_api_key \
  -e HF_TOKEN=your_hf_read_token \
  void-runpod:latest
```

The container proxies nginx on port `7862` to Gradio on port `7860`.

If you plan to use upstream `sam3` for quadmask generation, `HF_TOKEN` should be a Hugging Face token with read access from an account that has already been approved for the gated `facebook/sam3` repo.

## Runpod Deploy Notes

Expose ports `7862` and `8888` and launch on an A40 or L40 pod.

Environment variables:

- `VOID_USERNAME` basic-auth username, default `admin`
- `VOID_PASSWORD` basic-auth password, default `void`
- `GEMINI_API_KEY` Gemini API key used for quadmask generation
- `HF_TOKEN` Hugging Face token with read access. If you want `sam3` quadmask generation, the token must belong to an account that already has access to the gated `facebook/sam3` repo.

On first boot the startup script copies the app to `/workspace/VOID-on-Runpod`, downloads:

- `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`
- `netflix/void-model` `void_pass1.safetensors`
- `sam2.1_hiera_large.pt`

It also installs the git-based runtime Python packages needed by the upstream subtree:

- `facebookresearch/sam2` pinned in the Docker image to `aa9b8722d0585b661ded4b3dff1bd103540554ae` and installed with `SAM2_BUILD_CUDA=0 pip install -e .` for SAM 2.1 compatibility
- `facebookresearch/sam3`

The startup script assumes a SAM 2.1-only environment. `sam2` is expected to be preinstalled in the image.
The image intentionally skips the optional SAM2 CUDA post-processing extension for a more reliable build. The stage-1 pipeline therefore defaults `VOID_SAM2_APPLY_POSTPROCESSING=0`; set it to `1` only if you rebuild SAM2 with the extension enabled.

The Pass 2 checkpoint `void_pass2.safetensors` is downloaded lazily the first time a user runs Pass 2.

Important Hugging Face note:

- The base CogVideoX model and VOID checkpoints are fetched from Hugging Face with `HF_TOKEN` when provided.
- The official `sam3` Python package is installed from GitHub, but the official SAM3 weights are gated on Hugging Face.
- If you want the app to use `sam3` during quadmask generation, the Hugging Face account behind `HF_TOKEN` must already have been granted access to `facebook/sam3`.
- Without that approval, Pass 1 and Pass 2 can still run from an existing `quadmask_0.mp4`, but new `sam3`-backed quadmask generation will not work.

The default basic-auth credentials are:

- username: `admin`
- password: `void`

The Gradio app then guides the user through:

1. Uploading a video and optionally assigning a reusable job name.
2. Clicking points on any frame to build the original points JSON.
3. Setting the clean-background prompt, with Gemini auth coming from `GEMINI_API_KEY` in the pod environment.
4. Creating and previewing `quadmask_0.mp4`.
5. Running Pass 1.
6. Running Pass 2.

If you assign a job name, the UI shows the exact saved value as `Active job`. You can paste that later into `Existing job name` to reopen the same job workspace without reuploading.

If you already have intermediate assets, you can also:

- upload an existing `quadmask_0.mp4`
- upload an existing Pass 1 output video and continue directly to Pass 2

The interactive point-selection stage uses the upstream SAM2 flow from the `VOID` subtree. The grey-mask stage remains the upstream text-conditioned helper path behind the scenes, but that is not exposed in the web UI.

## Output Layout

The app writes generated job data into the workspace directory instead of the repo root.

Local default:

- `.void-workspace/`

Runpod default:

- `/workspace/VOID-on-Runpod`

Per uploaded job, outputs are written under:

```text
<workspace>/jobs/<job-id>/
├── data/sequence/
│   ├── input_video.mp4
│   ├── prompt.json
│   ├── black_mask.mp4
│   ├── grey_mask.mp4
│   ├── quadmask_0.mp4
│   └── vlm_analysis.json
├── pass1_outputs/
│   └── sequence-fg=-1-*.mp4
├── pass2_outputs/
│   └── sequence_warped_noise_inference.mp4
└── logs/
    ├── quadmask.log
    ├── pass1.log
    └── pass2.log
```

The local workspace directory `.void-workspace/` is gitignored.

## Logs

Inside the pod:

```bash
tail -f /workspace/VOID-on-Runpod/logs/void.log
```

## Restart

Inside the pod:

```bash
restart-void
```
