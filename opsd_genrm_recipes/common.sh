#!/bin/bash
# Shared helpers for the OPSD-GenRM launchers. Sourced by the scripts under
# opsd_genrm_recipes/qwen3_*/; not meant to be executed directly.
#
# Environment variables (all optional):
#   DATA_ROOT         Preprocessed parquet output dir   (default: <repo>/datasets)
#   CKPT_ROOT         Checkpoint output dir             (default: <repo>/checkpoints)
#   WANDB_PROJECT     W&B project name                  (default: OPSD-GenRM)
#   N_GPUS_PER_NODE   GPUs per node                     (default: 8)
#   SKIP_PREPROCESS   Set to 1 to reuse existing parquet files
#   SKIP_MERGE        Set to 1 to skip FSDP -> HF checkpoint conversion
#
# Credentials are read from opsd_genrm_recipes/credentials.env (see
# credentials.env.example) or from the environment.

OPSD_RECIPES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${OPSD_RECIPES_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints}"
WANDB_PROJECT="${WANDB_PROJECT:-OPSD-GenRM}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

opsd_init() {
    DRY_RUN=false
    if [[ "$1" == "--dry-run" ]]; then
        DRY_RUN=true
        echo "Dry run: commands are printed but not executed."
    fi

    export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${RANK:-0}"
    unset VLLM_ATTENTION_BACKEND
    export VLLM_USE_V1=1
    export PYTHONUNBUFFERED=1
    ulimit -c 0

    opsd_load_credentials
}

opsd_load_credentials() {
    local cred_file="${OPSD_CREDENTIALS_FILE:-${OPSD_RECIPES_DIR}/credentials.env}"
    if [[ -f "$cred_file" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "$cred_file"
        set +a
    fi

    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        LOGGER="['console','wandb']"
    else
        echo "WANDB_API_KEY not set; logging to console only."
        LOGGER="['console']"
    fi

    # Empty values would override cached logins, so drop them.
    [[ -z "${HF_TOKEN:-}" ]] && unset HF_TOKEN
    [[ -z "${WANDB_ENTITY:-}" ]] && unset WANDB_ENTITY
}

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

# Builds the HelpSteer3 training set and the RM-Bench and RewardBench 2
# evaluation sets under ${DATA_ROOT}/<prompt template>/, since the rendered
# prompts depend on the template. Sets TRAIN_DATA_DIR and VAL_FILES.
#   $1: prompt template (pair_rm | pair_rm_rubric)
#   $2: teacher feedback mode (reasoning | rubric | none)
opsd_preprocess() {
    local prompt_template="$1"
    local feedback_mode="$2"
    local data_dir="${DATA_ROOT}/${prompt_template}"

    TRAIN_DATA_DIR="${data_dir}/helpsteer3_dedup_${feedback_mode}"
    VAL_FILES="['${TRAIN_DATA_DIR}/test.parquet','${data_dir}/rmbench/test.parquet','${data_dir}/rewardbench2/test.parquet']"

    if [[ "${SKIP_PREPROCESS:-0}" == "1" || "$DRY_RUN" == true ]]; then
        return 0
    fi

    echo "Preprocessing data into ${data_dir} ..."
    (
        set -e
        cd "$REPO_ROOT"
        python data/preprocess_helpsteer3_dedup.py \
            --local_dir "$TRAIN_DATA_DIR" \
            --prompt_template "$prompt_template" \
            --feedback_mode "$feedback_mode" \
            --include_multiturn \
            --include_format
        python data/preprocess_rmbench.py \
            --local_dir "${data_dir}/rmbench" \
            --prompt_template "$prompt_template" \
            --include_format
        python data/preprocess_rewardbench2.py \
            --local_dir "${data_dir}/rewardbench2" \
            --prompt_template "$prompt_template" \
            --include_format
    ) || { echo "ERROR: data preprocessing failed." >&2; exit 1; }
}

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

# Runs `python -m verl.trainer.main_ppo "$@"`, teeing output to
# ${CKPT_DIR}/train.log. Sets TRAIN_EXIT.
opsd_train() {
    local cmd=(python -m verl.trainer.main_ppo "$@")

    if [[ "$DRY_RUN" == true ]]; then
        echo "Experiment: ${EXP_NAME}"
        printf '%q ' "${cmd[@]}"
        echo
        TRAIN_EXIT=1
        return 0
    fi

    mkdir -p "$CKPT_DIR"
    echo "================================================================"
    echo "Experiment: ${EXP_NAME}"
    echo "Checkpoints: ${CKPT_DIR}"
    echo "================================================================"
    cd "$REPO_ROOT"
    set -o pipefail
    "${cmd[@]}" 2>&1 | tee "${CKPT_DIR}/train.log"
    TRAIN_EXIT=$?
    set +o pipefail
}

# Converts every saved FSDP actor checkpoint in ${CKPT_DIR} to Hugging Face
# format under ${CKPT_DIR}/hf_step_<N>. On multi-node runs without a shared
# filesystem, shards are first collected from WORKERS onto this node.
opsd_merge_checkpoints() {
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    if [[ "${TRAIN_EXIT:-1}" -ne 0 ]]; then
        return "${TRAIN_EXIT:-1}"
    fi
    if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
        return 0
    fi

    local steps
    steps=$(
        {
            ls -d "${CKPT_DIR}"/global_step_*/ 2>/dev/null
            for w in "${WORKERS[@]}"; do
                ssh "$w" "ls -d ${CKPT_DIR}/global_step_*/ 2>/dev/null" || true
            done
        } | sed -nE 's|.*/global_step_([0-9]+)/?$|\1|p' | sort -un
    )

    for step in $steps; do
        local actor_dir="${CKPT_DIR}/global_step_${step}/actor"
        local target_dir="${CKPT_DIR}/hf_step_${step}"
        mkdir -p "$actor_dir"

        # --ignore-existing makes this a no-op on a shared filesystem.
        for w in "${WORKERS[@]}"; do
            rsync -a --ignore-existing "${w}:${actor_dir}/" "${actor_dir}/" 2>/dev/null || true
        done

        # The merger needs a local HF config; fall back to the base model's.
        if [[ ! -f "${actor_dir}/huggingface/config.json" ]]; then
            python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODEL_PATH}', local_dir='${actor_dir}/huggingface', \
allow_patterns=['*.json', '*.py', '*.txt', 'tokenizer*'])"
        fi

        echo "Converting ${actor_dir} -> ${target_dir}"
        (cd "$REPO_ROOT" && python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$actor_dir" \
            --target_dir "$target_dir") \
            || echo "WARNING: conversion failed for step ${step}; FSDP shards kept in ${actor_dir}." >&2
    done
}

# -----------------------------------------------------------------------------
# Multi-node (Ray over SSH)
# -----------------------------------------------------------------------------
#
# Requires passwordless SSH from the head node to every worker, the same
# Python environment on every node, and the repo at the same absolute path.
#   WORKER_NODE_IPS   Comma-separated worker addresses (required)
#   HEAD_NODE_IP      Head address reachable by workers (default: first IP of this host)
#   RAY_PORT          Ray GCS port (default: 6379)
#   SKIP_SYNC         Set to 1 if the repo is on a shared filesystem
#   NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME are forwarded to workers if set.

opsd_multinode_init() {
    if [[ -z "${WORKER_NODE_IPS:-}" ]]; then
        echo "ERROR: set WORKER_NODE_IPS to a comma-separated list of worker addresses." >&2
        exit 1
    fi
    IFS=',' read -ra WORKERS <<< "$WORKER_NODE_IPS"
    HEAD_NODE_IP="${HEAD_NODE_IP:-$(hostname -I | awk '{print $1}')}"
    RAY_PORT="${RAY_PORT:-6379}"
    NNODES=$(( 1 + ${#WORKERS[@]} ))
}

# Mirrors the repo (code + preprocessed data) from the head node to workers.
opsd_sync_workers() {
    if [[ "${SKIP_SYNC:-0}" == "1" || "$DRY_RUN" == true ]]; then
        return 0
    fi
    echo "Syncing ${REPO_ROOT} to ${#WORKERS[@]} worker(s) ..."
    for w in "${WORKERS[@]}"; do
        (
            ssh "$w" "mkdir -p '${REPO_ROOT}' '${DATA_ROOT}'" &&
                rsync -a --delete \
                    --exclude '.git' --exclude '__pycache__' --exclude 'wandb' --exclude 'outputs' \
                    --exclude "/$(basename "$CKPT_ROOT")/" \
                    "${REPO_ROOT}/" "${w}:${REPO_ROOT}/" &&
                rsync -a "${DATA_ROOT}/" "${w}:${DATA_ROOT}/"
        ) || echo "WARNING: sync to ${w} failed." >&2 &
    done
    wait
}

opsd_start_ray_cluster() {
    if [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    # Environment forwarded to Ray workers; the head inherits this shell's env.
    local fwd="" var
    for var in WANDB_API_KEY WANDB_ENTITY HF_TOKEN NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME; do
        [[ -n "${!var:-}" ]] && fwd+="${var}=$(printf '%q' "${!var}") "
    done

    echo "Starting Ray: head=${HEAD_NODE_IP}, workers=${WORKERS[*]}"
    ray stop --force >/dev/null 2>&1 || true
    for w in "${WORKERS[@]}"; do
        ssh "$w" "ray stop --force" >/dev/null 2>&1 || true
    done

    # FSDP CPU offload keeps host memory high; raise Ray's OOM-kill threshold.
    local ray_opts="--num-gpus=${N_GPUS_PER_NODE} --object-store-memory=20000000000"
    RAY_memory_usage_threshold=0.98 ray start --head --port="$RAY_PORT" $ray_opts \
        || { echo "ERROR: failed to start Ray head." >&2; exit 1; }
    for w in "${WORKERS[@]}"; do
        ssh "$w" "cd '${REPO_ROOT}' && env ${fwd}RAY_memory_usage_threshold=0.98 ray start --address=${HEAD_NODE_IP}:${RAY_PORT} ${ray_opts}" \
            || { echo "ERROR: failed to start Ray on ${w}." >&2; exit 1; }
    done
    sleep 5
    ray status
}
