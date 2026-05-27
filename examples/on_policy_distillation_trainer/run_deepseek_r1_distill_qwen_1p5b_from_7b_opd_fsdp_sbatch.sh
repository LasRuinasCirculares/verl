#!/usr/bin/env bash
# On-policy distillation | DeepSeek-R1-Distill-Qwen 1.5B <- 7B | vLLM rollout | FSDP | Slurm 8 GPUs
# Submit from verl-main with:
#   sbatch examples/on_policy_distillation_trainer/run_deepseek_r1_distill_qwen_1p5b_from_7b_opd_fsdp_sbatch.sh

#SBATCH --job-name=opd_ds1p5b_7b
#SBATCH --chdir=/mnt/data1/zhangjun2025/Lineage/verl-main
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=test
#SBATCH --partition=gpu
##SBATCH --exclude=g[81-82]
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

TOTAL_GPUS=${TOTAL_GPUS:-8}
NNODES=${NNODES:-1}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TRAINER_NGPUS_PER_NODE=${TRAINER_NGPUS_PER_NODE:-4}
TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-4}
REQUIRED_GPUS=$((TRAINER_NGPUS_PER_NODE * NNODES + TEACHER_NGPUS_PER_NODE * TEACHER_NNODES))
if [ "$REQUIRED_GPUS" -ne "$TOTAL_GPUS" ]; then
    echo "GPU resource mismatch: trainer ${TRAINER_NGPUS_PER_NODE}*${NNODES} + teacher ${TEACHER_NGPUS_PER_NODE}*${TEACHER_NNODES} = ${REQUIRED_GPUS}, but TOTAL_GPUS=${TOTAL_GPUS}."
    echo "Keep the sum equal to the 8 GPUs requested by this sbatch script, or override TOTAL_GPUS consistently."
    exit 1
fi

if [ -n "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES"
elif [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi
echo "Ray visible GPUs: ${CUDA_VISIBLE_DEVICES:-all Slurm-visible GPUs}; TOTAL_GPUS=${TOTAL_GPUS}; trainer GPUs=${TRAINER_NGPUS_PER_NODE}; teacher GPUs=${TEACHER_NGPUS_PER_NODE}"

STUDENT_MODEL=${STUDENT_MODEL:-${LINEAGE_ROOT}/models/DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-${LINEAGE_ROOT}/models/JustRL-DeepSeek-1.5B}

TRAIN_FILE=${TRAIN_FILE:-${LINEAGE_ROOT}/Training_Dynamic-main/datasets/dapo-math-17k-processed.parquet}
TEST_DATA_DIR=${TEST_DATA_DIR:-${LINEAGE_ROOT}/dataset}
TRAIN_FILES=${TRAIN_FILES:-"['$TRAIN_FILE']"}
VAL_FILES=${VAL_FILES:-"['$TEST_DATA_DIR/Math/math500.parquet', '$TEST_DATA_DIR/Math/aime.parquet', '$TEST_DATA_DIR/General/gpqa.parquet']"}
TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-full-7168}

case "$STUDENT_MODEL" in
    /*) [ -d "$STUDENT_MODEL" ] || { echo "Student model path does not exist: $STUDENT_MODEL"; exit 1; } ;;
esac
case "$TEACHER_MODEL" in
    /*) [ -d "$TEACHER_MODEL" ] || { echo "Teacher model path does not exist: $TEACHER_MODEL"; exit 1; } ;;
esac
case "$TRAIN_FILE" in
    /*) [ -f "$TRAIN_FILE" ] || { echo "Training file does not exist: $TRAIN_FILE"; exit 1; } ;;
esac

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-${MAX_RESP_LENGTH:-7168}}
MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-${MAX_VAL_RESP_LENGTH:-7168}}
MAX_TRAIN_SEQUENCE_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_VAL_SEQUENCE_LEN=$((MAX_PROMPT_LENGTH + MAX_VAL_RESPONSE_LENGTH))
MAX_SEQUENCE_LEN=$((MAX_TRAIN_SEQUENCE_LEN > MAX_VAL_SEQUENCE_LEN ? MAX_TRAIN_SEQUENCE_LEN : MAX_VAL_SEQUENCE_LEN))
MIN_DISTILLATION_MODEL_LEN=$((MAX_SEQUENCE_LEN + 1))
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$MIN_DISTILLATION_MODEL_LEN}
if [ "$MAX_MODEL_LEN" -lt "$MIN_DISTILLATION_MODEL_LEN" ]; then
    echo "MAX_MODEL_LEN=${MAX_MODEL_LEN} is too small for distillation; need at least prompt + response + 1 = ${MIN_DISTILLATION_MODEL_LEN}."
    exit 1
fi
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_MODEL_LEN > 32768 ? MAX_MODEL_LEN : 32768))}

MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$MINI_BATCH_SIZE}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$MINI_BATCH_SIZE}
TEMPERATURE=${TEMPERATURE:-1.0}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
if [ "$TEACHER_TEMPERATURE" != "1.0" ] && [ "$TEACHER_TEMPERATURE" != "1" ]; then
    echo "Current verl-main teacher prompt-logprob path requires TEACHER_TEMPERATURE=1.0; got ${TEACHER_TEMPERATURE}."
    exit 1
fi
N_RESPONSES=${N_RESPONSES:-16}
VAL_N=${VAL_N:-8}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
ACTOR_LR=${ACTOR_LR:-${LR:-1e-6}}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
USE_KL=${USE_KL:-False}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$PPO_MAX_TOKEN_LEN_PER_GPU}
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.4}
TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-$PPO_MAX_TOKEN_LEN_PER_GPU}

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-${LOG_PROB_TOP_K:-16}}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}

PROJECT_NAME=${PROJECT_NAME:-OnPolicyDistillation}
STUDENT_MODEL_NAME=$(basename "$STUDENT_MODEL")
TEACHER_MODEL_NAME=$(basename "$TEACHER_MODEL")
EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_${TRAIN_DATASET_NAME}_${STUDENT_MODEL_NAME}_${TEACHER_MODEL_NAME}_${MAX_RESPONSE_LENGTH}-T_${TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-dl_${DISTILLATION_LOSS_MODE}-topk_${DISTILLATION_TOPK}}
PROJECT_PATH=${PROJECT_PATH:-checkpoint}
CKPT_PATH=${CKPT_PATH:-${PROJECT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}}
VALIDATION_LOG_DIR=${VALIDATION_LOG_DIR:-validation_log/${EXPERIMENT_NAME}}

TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
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

RESUME_FROM_PATH=${RESUME_FROM_PATH:-null}
if [ -z "${RESUME_MODE+x}" ]; then
    if [ "$RESUME_FROM_PATH" != "null" ]; then
        RESUME_MODE=resume_path
    elif [ "${OPD_NEW_RUN:-0}" = "1" ]; then
        RESUME_MODE=disable
    else
        RESUME_MODE=auto
    fi
fi

KL_ARGS=()
if [ "$USE_KL" = "True" ] || [ "$USE_KL" = "true" ]; then
    KL_ARGS=(
        actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.actor.kl_loss_coef=0.005
        actor_rollout_ref.actor.kl_loss_type=low_var_kl
    )
else
    KL_ARGS=(actor_rollout_ref.actor.use_kl_loss=False)
fi

LR_ARGS=()
if [ "$LR_SCHEDULER" = "cosine" ]; then
    LR_ARGS=(
        actor_rollout_ref.actor.optim.warmup_style=cosine
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03
    )
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

DATA=(
    algorithm.adv_estimator=${ADV_ESTIMATOR}
    algorithm.use_kl_in_reward=False
    data.shuffle=False
    data.return_raw_chat=True
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}
    "${KL_ARGS[@]}"
    "${LR_ARGS[@]}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.n=${N_RESPONSES}
    actor_rollout_ref.rollout.temperature=${TEMPERATURE}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
)

REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger="$TRAINER_LOGGER"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${TRAINER_NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=True
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.validation_data_dir=${VALIDATION_LOG_DIR}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.resume_mode=${RESUME_MODE}
    trainer.resume_from_path=${RESUME_FROM_PATH}
    trainer.default_local_dir=${CKPT_PATH}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.temperature=${TEACHER_TEMPERATURE}
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_NUM_BATCHED_TOKENS}
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
