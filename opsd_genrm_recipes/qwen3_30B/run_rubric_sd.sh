#!/bin/bash
# OPSD-GenRM on Qwen3-30B-A3B-Instruct-2507.
# Teacher feedback: task rubric; no token mask.
#
# Hardware: 4 nodes x 8 GPUs (Ray over SSH).
# Usage:    WORKER_NODE_IPS=<ip1>,<ip2>,<ip3> bash opsd_genrm_recipes/qwen3_30B/run_rubric_sd.sh [--dry-run]
# Run on the head node. See opsd_genrm_recipes/common.sh for multi-node options.

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
opsd_init "$@"
opsd_multinode_init

MODEL_PATH="Qwen/Qwen3-30B-A3B-Instruct-2507"
CONFIG_NAME=opsd_genrm_rubric
SEED=42

# Optimization
TRAIN_BATCH_SIZE=256
EPOCH=3
LR=1e-5
LOSS_AGG_MODE=token-mean

# Rollout
ROLLOUT_N=4
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=8192
MAX_MODEL_LEN=12800

# Self-distillation
PROMPT_TEMPLATE=pair_rm_rubric
FEEDBACK_MODE=rubric
MAX_REPROMPT_LEN=5120
ALPHA=1
EMA_UPDATE_RATE=0.01
DISTILLATION_TOPK=100
TOKEN_MASK_MODE=none
TOKEN_MASK_TOP_PCT=100

EXP_NAME="${MODEL_PATH#*/}-sd-${FEEDBACK_MODE}-${TOKEN_MASK_MODE}${TOKEN_MASK_TOP_PCT}-lr${LR}-seed${SEED}"
CKPT_DIR="${CKPT_ROOT}/${EXP_NAME}"

opsd_preprocess "$PROMPT_TEMPLATE" "$FEEDBACK_MODE"
opsd_sync_workers
opsd_start_ray_cluster

opsd_train --config-name "$CONFIG_NAME" \
    data.train_files="['${TRAIN_DATA_DIR}/train.parquet']" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.shuffle=True \
    data.seed=${SEED} \
    data.swap_order_per_epoch=True \
    data.trust_remote_code=True \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN} \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=full_logit_kl \
    actor_rollout_ref.actor.self_distillation.enable=True \
    actor_rollout_ref.actor.self_distillation.prompt_template=${PROMPT_TEMPLATE} \
    actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK} \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${EMA_UPDATE_RATE} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
    actor_rollout_ref.actor.self_distillation.include_solution=False \
    actor_rollout_ref.actor.self_distillation.include_answer=False \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LEN} \
    actor_rollout_ref.actor.self_distillation.token_mask_mode=${TOKEN_MASK_MODE} \
    actor_rollout_ref.actor.self_distillation.token_mask_top_pct=${TOKEN_MASK_TOP_PCT} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    reward.custom_reward_function.path="${REPO_ROOT}/verl/utils/reward_score/feedback/__init__.py" \
    critic.model.path="${MODEL_PATH}" \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs=${EPOCH} \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=6

opsd_merge_checkpoints
