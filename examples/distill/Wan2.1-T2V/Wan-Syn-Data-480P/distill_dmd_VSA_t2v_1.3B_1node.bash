#!/bin/bash
set -e -x

# 1. 环境设置 (Environment Setup)
# 请确保已经正确安装了 miniconda 并且存在名为 fastvideo 的环境
# 如果你的 anaconda/miniconda 不在此路径，请修改下面的 source 命令
source /path/to/miniconda/bin/activate
conda activate fastvideo

# 2. 基本信息 (Basic Info)
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# 为不同进程设置不同的缓存目录
export TRITON_CACHE_DIR=/tmp/triton_cache_${RANK}
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# 为单机运行设置分布式环境
export MASTER_ADDR=localhost
export MASTER_PORT=29500 # 你可以选择任何一个空闲端口
export NODE_RANK=0

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# 3. 配置 (Configs)
# !! 根据你的单机 GPU 数量修改此值
NUM_GPUS=8

MODEL_PATH="/path/to/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REAL_SCORE_MODEL_PATH="/path/to/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
FAKE_SCORE_MODEL_PATH="/path/to/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR="/path/to/Wan-Syn_77x448x832_600k/train"
VALIDATION_DATASET_FILE="/path/to/Wan-Syn_77x448x832_600k/val/Part_1/latents_chunk_0000.parquet"
OUTPUT_DIR="checkpoints/distill_dmd_cm_vsa_wan_t2v_1.3b"

# 可选: 如果你想指定使用哪几块GPU，可以取消下面这行的注释
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 4. 参数数组 (Argument Arrays)

# 训练参数
training_args=(
  --tracker_project_name wan_t2v_distill_dmd_VSA
  --output_dir $OUTPUT_DIR
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --num_frames 81
  --enable_gradient_checkpointing_type "full"
)

# 并行参数
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim $NUM_GPUS
  --hsdp_shard_dim 1
)

# 模型参数
model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

# 数据集参数
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# 验证参数
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
  --validation_sampling_steps "3"
  --validation_guidance_scale "6.0" # not used for dmd inference
)

# 优化器参数
optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# 其他参数
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 8
  --seed 1000
)

# DMD 参数
dmd_args=(
  --dmd_denoising_steps '1000,757,522'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --generator_update_interval 5
  --real_score_guidance_scale 3.5
  --VSA_sparsity 0.8
)


# 5. 启动训练 (Launch Training)
# 使用 torchrun 在单机上启动多GPU训练
torchrun \
--nnodes 1 \
--nproc_per_node $NUM_GPUS \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wan_meanflow_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"