cd "$(dirname "$0")/../.."

export MODEL_NAME="./CogVideoX-Fun-V1.5-5b-InP"
export DATASET_NAME=""
export DATASET_META_NAME="datasets/void_train_data.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# Fix for shared memory issue
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
ulimit -n 65536 2>/dev/null || true

accelerate launch --num_processes=8 --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard scripts/cogvideox_fun/train_warped_noise.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=1 \
  --video_sample_n_frames=72 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --num_train_epochs=40 \
  --checkpointing_steps=2000 \
   --transformer_path="${TRANSFORMER_PATH:?'TRANSFORMER_PATH must be set to your stage 1 checkpoint'}" \
  --learning_rate=1e-05 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=50 \
  --seed=42 \
  --output_dir="${OUTPUT_DIR:-void_warped_noise_output}" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --random_frame_crop \
  --enable_bucket \
  --use_came \
  --use_deepspeed \
  --train_mode="void" \
  --resume_from_checkpoint="latest" \
  --trainable_modules "." \
  --use_vae_mask \
  --use_quadmask \
  --use_warped_noise \
  --warped_noise_degradation=0.3 \
  --warped_noise_probability=1.0

