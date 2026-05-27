#!/usr/bin/env bash
# On-policy distillation | DeepSeek-R1-Distill-Qwen 1.5B <- 7B | FSDP | Slurm 4 GPUs
# Submit from verl-main with:
#   sbatch examples/on_policy_distillation_trainer/run_deepseek_r1_distill_qwen_1p5b_from_7b_opd_fsdp_4gpu_sbatch.sh

#SBATCH --job-name=opd_ds1p5b_7b_4g
#SBATCH --chdir=/mnt/data1/zhangjun2025/Lineage/verl-main
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --account=test
#SBATCH --partition=gpu
##SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -xeuo pipefail

VERL_ROOT=${VERL_ROOT:-/mnt/data1/zhangjun2025/Lineage/verl-main}
SCRIPT_DIR=${OPD_SCRIPT_DIR:-${VERL_ROOT}/examples/on_policy_distillation_trainer}
MAIN_SCRIPT="${SCRIPT_DIR}/run_deepseek_r1_distill_qwen_1p5b_from_7b_opd_fsdp_sbatch.sh"

export TOTAL_GPUS=${TOTAL_GPUS:-4}
export TRAINER_NGPUS_PER_NODE=${TRAINER_NGPUS_PER_NODE:-2}
export TEACHER_NGPUS_PER_NODE=${TEACHER_NGPUS_PER_NODE:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export TEACHER_TP=${TEACHER_TP:-2}

if [ -z "${SLURM_JOB_ID:-}" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${TRAIN_CUDA_VISIBLE_DEVICES:-}" ]; then
    export TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3
fi

MAX_RESPONSE_LENGTH_FOR_NAME=${MAX_RESPONSE_LENGTH:-${MAX_RESP_LENGTH:-7168}}
TEMPERATURE_FOR_NAME=${TEMPERATURE:-1.0}
N_RESPONSES_FOR_NAME=${N_RESPONSES:-4}
MINI_BATCH_SIZE_FOR_NAME=${MINI_BATCH_SIZE:-64}
DISTILLATION_LOSS_MODE_FOR_NAME=${DISTILLATION_LOSS_MODE:-k1}
DISTILLATION_TOPK_FOR_NAME=${DISTILLATION_TOPK:-${LOG_PROB_TOP_K:-16}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opd_4g_DAPO-Math-17k-7168_R1-1.5B_JustRL-1.5B_${MAX_RESPONSE_LENGTH_FOR_NAME}-T_${TEMPERATURE_FOR_NAME}-n_${N_RESPONSES_FOR_NAME}-mbs_${MINI_BATCH_SIZE_FOR_NAME}-dl_${DISTILLATION_LOSS_MODE_FOR_NAME}-topk_${DISTILLATION_TOPK_FOR_NAME}}

if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "Main OPD sbatch script does not exist: $MAIN_SCRIPT"
    exit 1
fi

exec bash "$MAIN_SCRIPT" "$@"
