# VOID RunPod Template (Unofficial)

### _This template has been tested with an A40_

This template launches a Gradio UI for [VOID](https://github.com/Netflix/void-model) on RunPod. It is set up for the interactive VOID workflow:

- point-based primary-object selection
- Gemini-backed quadmask generation
- Pass 1 inference
- optional Pass 2 refinement

## Quick Start

1. Get a Gemini API key with billing turned on: https://aistudio.google.com/api-keys
2. Get approval to access the SAM3 model weights on Hugging Face: https://huggingface.co/facebook/sam3
3. Create a Hugging Face token with read access: https://huggingface.co/settings/tokens
4. Deploy the template on an `A40` or `L40`.
5. Set these environment variables on the pod:
   - `VOID_USERNAME`
   - `VOID_PASSWORD`
   - `GEMINI_API_KEY`
   - `HF_TOKEN`
6. Open the app on port `7862` and log in. If you did not change the username and password they are `admin` and `void` respectively.

## Template Variables

| Variable         | Description                                                  | Default |
| ---------------- | ------------------------------------------------------------ | ------- |
| `VOID_USERNAME`  | nginx / Gradio basic-auth username                           | `admin` |
| `VOID_PASSWORD`  | nginx / Gradio basic-auth password                           | `void`  |
| `GEMINI_API_KEY` | Gemini API key used for quadmask generation                  | none    |
| `HF_TOKEN`       | Hugging Face token for model downloads and gated SAM3 access | none    |

## Models And Requirements

The template downloads and uses:

- `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`
- `netflix/void-model` `void_pass1.safetensors`
- `sam2.1_hiera_large.pt`

## Outputs And Logs

- Main app log: `/workspace/VOID-on-Runpod/logs/void.log`
- Per-job logs: `/workspace/VOID-on-Runpod/jobs/<job-id>/logs`
- Outputs are stored under `/workspace/VOID-on-Runpod/jobs/<job-id>`

Tail logs inside the pod:

```bash
tail -f /workspace/VOID-on-Runpod/logs/void.log
```

Restart the app inside the pod:

```bash
restart-void
```

## Notes

- `void_pass2.safetensors` is downloaded only when Pass 2 is used.
- SAM3 is used for grey-mask generation and requires Hugging Face access to `facebook/sam3`.
- Gemini billing must be enabled for quadmask generation to work.
- This template wraps the upstream VOID subtree in a RunPod-focused UI and Docker image.
- It uses SAM 2.1 for Stage 1 primary-object segmentation.
- It keeps affected regions more continuous over time when Gemini only identifies them on a few frames.
- It includes a small SAM3 runtime wrapper to avoid dtype and fused-kernel failures seen in this environment.
- Recommended GPU class: A40 or L40.

## Resources

- The [Dockerfile and code](https://github.com/Square-Zero-Labs/VOID-on-Runpod) are public.
- Upstream VOID repo: https://github.com/Netflix/void-model
