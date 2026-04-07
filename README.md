# VOID on Runpod

This repository wraps the upstream [`VOID`](./VOID) subtree in a Runpod-friendly Gradio app and Docker image.

The app supports:

- browser-based point selection across arbitrary frames
- Gemini-backed quadmask generation
- staged Pass 1 and Pass 2 execution
- reopening an existing saved job by name without reuploading the source video
- uploading an existing Pass 1 result and continuing directly to Pass 2
- first-boot download of the required models and SAM2 checkpoint
- lazy download of the Pass 2 checkpoint only when Pass 2 is requested
- nginx basic auth in front of Gradio

## Build

```bash
docker build --platform=linux/amd64 -t void-runpod:latest .
```

## Run Locally

```bash
docker run --gpus all -p 7862:7862 \
  -e VOID_USERNAME=admin \
  -e VOID_PASSWORD=void \
  -e HF_TOKEN=your_hf_token \
  void-runpod:latest
```

The container proxies nginx on port `7862` to Gradio on port `7860`.

## Runpod Deploy Notes

Expose port `7862` and launch on an A40 or L40 pod.

Optional environment variables:

- `VOID_USERNAME` basic-auth username, default `admin`
- `VOID_PASSWORD` basic-auth password, default `void`
- `HF_TOKEN` optional Hugging Face token for model downloads
- `VOID_WORKSPACE_DIR` override for where jobs and checkpoints are stored

On first boot the startup script copies the app to `/workspace/VOID-on-Runpod`, downloads:

- `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`
- `netflix/void-model` `void_pass1.safetensors`
- `sam2_hiera_large.pt`

It also installs the git-based runtime Python packages needed by the upstream subtree:

- `facebookresearch/segment-anything-2`
- `luca-medeiros/lang-segment-anything`

The Pass 2 checkpoint `void_pass2.safetensors` is downloaded lazily the first time a user runs Pass 2.

The default basic-auth credentials are:

- username: `admin`
- password: `void`

The Gradio app then guides the user through:

1. Uploading a video and optionally assigning a reusable job name.
2. Clicking points on any frame to build the original points JSON.
3. Entering the Gemini API key and clean-background prompt.
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
