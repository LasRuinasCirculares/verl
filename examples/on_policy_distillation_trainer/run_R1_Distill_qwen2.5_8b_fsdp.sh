#!/usr/bin/env bash
# On-policy distillation | Qwen3-8B <- Qwen3-32B | vLLM rollout | FSDP | Slurm 8 GPUs
# Submit from verl-main with:
#   sbatch examples/on_policy_distillation_trainer/run_R1_Distill_qwen2.5_8b_fsdp.sh

#SBATCH --job-name=opd_qwen3_8b_32b
#SBATCH --chdir=/mnt/data1/zhangjun2025/Lineage/verl-main
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -xeuo pipefail

VERL_ROOT=${VERL_ROOT:-/mnt/data1/zhangjun2025/Lineage/verl-main}
LINEAGE_ROOT=${LINEAGE_ROOT:-/mnt/data1/zhangjun2025/Lineage}
cd "$VERL_ROOT"
mkdir -p logs

export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export RAY_DISABLE_USAGE_STATS=1
export RAY_memory_usage_threshold=0.99
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES

CONDA_ENV_NAME=${CONDA_ENV_NAME:-verl}
if [ -f "/ceph_home/zhangjun2025/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/ceph_home/zhangjun2025/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
else
    echo "conda was not found; continuing with the current Python environment."
fi

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/data1/zhangjun2025/Lineage/models/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/data1/zhangjun2025/Lineage/models/DeepSeek-R1-Distill-Qwen-7B}

TOTAL_GPUS=${TOTAL_GPUS:-8}
NNODES=${NNODES:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TRAINER_NGPUS_PER_NODE=${TRAINER_NGPUS_PER_NODE:-${NGPUS_PER_NODE:-4}}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-${TEACHER_WORLD_SIZE:-4}}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$TRAINER_NGPUS_PER_NODE}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-$TEACHER_NGPUS_PER_NODE}

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

ACTOR_LR=${ACTOR_LR:-1e-6}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.4}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-200}
TEST_FREQ=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-OnPolicyDistillation}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_from_qwen3_32b_vllm_fsdp-rollout16}
PROJECT_PATH=${PROJECT_PATH:-checkpoint}
CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
VALIDATION_LOG_DIR=${VALIDATION_LOG_DIR:-validation_log/${EXPERIMENT_NAME}}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
# ---- end user-adjustable ----

REQUIRED_GPUS=$((TRAINER_NGPUS_PER_NODE * NNODES + TEACHER_NGPUS_PER_NODE * TEACHER_NNODES))
if [ "$REQUIRED_GPUS" -ne "$TOTAL_GPUS" ]; then
    echo "GPU resource mismatch: trainer ${TRAINER_NGPUS_PER_NODE}*${NNODES} + teacher ${TEACHER_NGPUS_PER_NODE}*${TEACHER_NNODES} = ${REQUIRED_GPUS}, but TOTAL_GPUS=${TOTAL_GPUS}."
    echo "Keep the sum equal to the GPUs requested by this sbatch script, or override TOTAL_GPUS consistently."
    exit 1
fi

if [ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"
elif [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
echo "Ray visible GPUs: ${CUDA_VISIBLE_DEVICES:-all Slurm-visible GPUs}; TOTAL_GPUS=${TOTAL_GPUS}; trainer GPUs=${TRAINER_NGPUS_PER_NODE}; teacher GPUs=${TEACHER_NGPUS_PER_NODE}"

case "$STUDENT_MODEL" in
    /*) [ -d "$STUDENT_MODEL" ] || { echo "Student model path does not exist: $STUDENT_MODEL"; exit 1; } ;;
esac
case "$TEACHER_MODEL" in
    /*) [ -d "$TEACHER_MODEL" ] || { echo "Teacher model path does not exist: $TEACHER_MODEL"; exit 1; } ;;
esac

TRAIN_FILES_WAS_SET=${TRAIN_FILES+x}
VAL_FILES_WAS_SET=${VAL_FILES+x}
GSM8K_TRAIN=${GSM8K_TRAIN:-/mnt/data1/zhangjun2025/Lineage/Training_Dynamic-main/datasets/dapo-math-17k-processed.parquet}
GSM8K_TEST=${GSM8K_TEST:-/mnt/data1/zhangjun2025/Lineage/dataset/Math/math500.parquet}
# MATH_TRAIN=${MATH_TRAIN:-$HOME/data/math/train.parquet}
# MATH_TEST=${MATH_TEST:-$HOME/data/math/test.parquet}

TRAIN_FILES=${TRAIN_FILES:-"['$GSM8K_TRAIN']"}
VAL_FILES=${VAL_FILES:-"['$GSM8K_TEST']"}

if [ -z "${TRAIN_FILES_WAS_SET:-}" ]; then
    [ -f "$GSM8K_TRAIN" ] || { echo "Training file does not exist: $GSM8K_TRAIN"; exit 1; }
    # [ -f "$MATH_TRAIN" ] || { echo "Training file does not exist: $MATH_TRAIN"; exit 1; }
fi
if [ -z "${VAL_FILES_WAS_SET:-}" ]; then
    [ -f "$GSM8K_TEST" ] || { echo "Validation file does not exist: $GSM8K_TEST"; exit 1; }
    # [ -f "$MATH_TEST" ] || { echo "Validation file does not exist: $MATH_TEST"; exit 1; }
fi

MAX_NUM_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ROOT=${WANDB_ROOT:-${PROJECT_PATH}/wandb}
export WANDB_DIR=${WANDB_DIR:-${WANDB_ROOT}}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${WANDB_ROOT}/data}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${WANDB_ROOT}/cache}
export WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR:-${WANDB_ROOT}/artifacts}
mkdir -p "$PROJECT_PATH" "$CKPT_PATH" "$VALIDATION_LOG_DIR" "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

if [[ "$TRAINER_LOGGER" == *wandb* && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set; set it before sbatch if you want online W&B logging."
fi

JOB_ID_FOR_PORT=${SLURM_JOB_ID:-$$}
RAY_PORT_BASE=${RAY_PORT_BASE:-$((20000 + (JOB_ID_FOR_PORT % 4000) * 10))}
RAY_PORT=${RAY_PORT:-$RAY_PORT_BASE}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-$((RAY_PORT_BASE + 1))}
RAY_NODE_MANAGER_PORT=${RAY_NODE_MANAGER_PORT:-$((RAY_PORT_BASE + 2))}
RAY_OBJECT_MANAGER_PORT=${RAY_OBJECT_MANAGER_PORT:-$((RAY_PORT_BASE + 3))}
RAY_MIN_WORKER_PORT=${RAY_MIN_WORKER_PORT:-$((RAY_PORT_BASE + 100))}
RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-$((RAY_PORT_BASE + 199))}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-32}}

ray stop --force || true
trap 'ray stop --force || true' EXIT
ray start --head \
    --num-gpus="$TOTAL_GPUS" \
    --num-cpus="$RAY_NUM_CPUS" \
    --port="$RAY_PORT" \
    --dashboard-port="$RAY_DASHBOARD_PORT" \
    --node-manager-port="$RAY_NODE_MANAGER_PORT" \
    --object-manager-port="$RAY_OBJECT_MANAGER_PORT" \
    --min-worker-port="$RAY_MIN_WORKER_PORT" \
    --max-worker-port="$RAY_MAX_WORKER_PORT" \
    --disable-usage-stats
sleep 5

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=16
    actor_rollout_ref.rollout.max_model_len=${MAX_NUM_TOKENS}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="$TRAINER_LOGGER"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${TRAINER_NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.validation_data_dir=${VALIDATION_LOG_DIR}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir=${CKPT_PATH}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_NUM_TOKENS}
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
