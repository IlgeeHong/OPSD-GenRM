# OPSD-GenRM

OPSD-GenRM trains generative reward models (pairwise LLM judges) with **on-policy self-distillation (OPSD)**, built on [verl](https://github.com/volcengine/verl) `release/v0.7.1`.

For each preference pair, the policy judges which of two responses is better. The same model then acts as a **teacher** on an enriched prompt that additionally contains privileged information — the annotator's rationale or a task rubric — and scores the student's own response under that prompt. The **student** (the policy on the original prompt) is trained to match the teacher's token distribution with a top-k KL loss. The teacher is an exponential moving average (EMA) of the student. Optionally, the loss is restricted to the 30% of tokens with the lowest entropy difference ΔH = H_student − H_teacher, i.e. where the teacher is most uncertain relative to the student.

---

## Installation

```bash
git clone <this-repo> OPSD-GenRM
cd OPSD-GenRM
pip install -r requirements.txt -r requirements-cuda.txt
pip install -e .
```

`vllm`, `flash-attn`, and `triton` are required; the fused top-k kernel needs `triton >= 2.3`.

### Credentials

```bash
cp opsd_genrm_recipes/credentials.env.example opsd_genrm_recipes/credentials.env
# then fill in WANDB_API_KEY (optional; console logging otherwise) and HF_TOKEN (optional)
```

`credentials.env` is git-ignored. Variables exported in your shell are picked up as well.

---

## Data

| Split | Dataset | Preprocessor |
|---|---|---|
| Train / validation | HelpSteer3, deduplicated, with rubrics ([`opsd-genrm/dedup_filtered_HS3`](https://huggingface.co/datasets/opsd-genrm/dedup_filtered_HS3)) | `data/preprocess_helpsteer3_dedup.py` |
| Evaluation | [RM-Bench](https://huggingface.co/datasets/THU-KEG/RM-Bench) | `data/preprocess_rmbench.py` |
| Evaluation | [RewardBench 2](https://huggingface.co/datasets/allenai/reward-bench-2) | `data/preprocess_rewardbench2.py` |

The launchers run these scripts automatically. `--feedback_mode` selects the teacher's privileged information: `reasoning` (annotator rationale) or `rubric`.

---

## Reproducing the experiments

Each launcher preprocesses the data (into `datasets/`), trains, and converts every saved checkpoint to Hugging Face format (under `checkpoints/<experiment>/hf_step_<N>`).

| Launcher | Method | Teacher feedback | Token mask |
|---|---|---|---|
| `run_drgrpo.sh` | Dr. GRPO baseline | — | — |
| `run_rationale_sd.sh` | OPSD | annotator rationale | none |
| `run_rationale_sd_mask70.sh` | OPSD | annotator rationale | `entropy_diff_mask`, bottom 30% of ΔH |
| `run_rubric_sd.sh` | OPSD | task rubric | none |
| `run_rubric_sd_mask70.sh` | OPSD | task rubric | `entropy_diff_mask`, bottom 30% of ΔH |

**Qwen3-4B** (`opsd_genrm_recipes/qwen3_4B/`, 1 node x 8 GPUs):

```bash
bash opsd_genrm_recipes/qwen3_4B/run_rationale_sd_mask70.sh
```

**Qwen3-30B-A3B** (`opsd_genrm_recipes/qwen3_30B/`, 4 nodes x 8 GPUs). Run on the head node; the launcher syncs the repo to the workers over SSH and starts the Ray cluster. It requires passwordless SSH to each worker, the same Python environment on every node, and the repo at the same absolute path on every node.

```bash
WORKER_NODE_IPS=10.0.0.2,10.0.0.3,10.0.0.4 bash opsd_genrm_recipes/qwen3_30B/run_rationale_sd_mask70.sh
```

Pass `--dry-run` to print the training command without running it. Other options (`DATA_ROOT`, `CKPT_ROOT`, `WANDB_PROJECT`, `HEAD_NODE_IP`, `SKIP_SYNC`, ...) are documented at the top of `opsd_genrm_recipes/common.sh`.

---

## Code overview

All self-distillation code paths are gated behind `actor_rollout_ref.actor.self_distillation.enable=true`; with it disabled, the trainer behaves like verl v0.7.1. Only the FSDP backend is supported.

### New modules

| Path | Purpose |
|---|---|
| `verl/trainer/ppo/self_distillation/teacher_batch.py` | Builds the teacher prompts (student prompt + feedback, optionally a successful peer rollout or the answer) and the per-sample self-distillation mask. |
| `verl/trainer/ppo/self_distillation/loss.py` | `full_logit_kl` policy loss: top-k KL between student and teacher with a tail bucket, importance-ratio clipping, and the entropy-based token masks. |
| `verl/trainer/ppo/self_distillation/teacher_model.py` | EMA teacher update. |
| `verl/trainer/ppo/self_distillation/reward.py` | `sd` advantage estimator, used only with `policy_loss.loss_mode=vanilla`. |
| `verl/utils/kernel/topk_logprobs.py` | Fused Triton kernel for top-k log-probabilities without materializing the full log-softmax. |
| `verl/utils/reward_score/feedback/` | Pairwise verdict reward (`<verdict>A\|B</verdict>`) that also returns the annotator feedback or rubric for the teacher. |
| `verl/trainer/ppo/token_mask.py` | Teacher-free entropy token mask for GRPO-style losses. |
| `verl/trainer/config/opsd_genrm.yaml` | Hydra config with the teacher prompt template for the rationale launchers. |
| `verl/trainer/config/opsd_genrm_rubric.yaml` | Same, with a rubric-oriented teacher instruction, for the rubric launchers. |
| `tests/trainer/ppo/self_distillation/test_sd_loss.py` | Unit tests for the loss, token mask, advantage estimator, and teacher updates. |

### Changes to verl files

| Path | Change |
|---|---|
| `verl/workers/actor/dp_actor.py` | Teacher forward pass, top-k extraction in `_forward_micro_batch`, self-distillation loss path in `update_policy`, token-count-weighted gradient accumulation for `token-mean`. |
| `verl/workers/fsdp_workers.py` | `compute_teacher_outputs` and `update_ema_teacher`. |
| `verl/trainer/ppo/ray_trainer.py` | Builds the teacher batch, runs the teacher forward pass and EMA update, and skips the old-log-prob pass when training is on-policy. |
| `verl/trainer/main_ppo.py` | Colocates the reference model with the actor when an EMA teacher is used. |
| `verl/workers/config/actor.py`, `verl/trainer/config/actor/actor.yaml` | `SelfDistillationConfig` and the actor-level token mask options. |
| `verl/utils/dataset/rl_dataset.py`, `verl/trainer/config/data/legacy_data.yaml` | `swap_order_per_epoch`: swaps Assistant A/B on odd epochs. |
| `verl/trainer/ppo/metric_utils.py` | Macro-averaged validation metrics for RM-Bench and RewardBench 2, per-domain metrics, and the RewardBench 2 all-correct reduction. |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py`, rollout configs | `min_p`, `repetition_penalty`, and `presence_penalty` sampling options. |
| `verl/utils/metric/utils.py` | NaN-aware metric aggregation. |

---

## Configuration reference

Self-distillation options live under `actor_rollout_ref.actor.self_distillation` (`verl/workers/config/actor.py::SelfDistillationConfig`). The loss is selected with `actor_rollout_ref.actor.policy_loss.loss_mode=full_logit_kl`.

| Field | Default | Meaning |
|---|---|---|
| `enable` | `false` | Enables the teacher batch, teacher forward pass, and EMA teacher update. |
| `distillation_topk` | `100` | Number of teacher top-k tokens used in the KL. |
| `distillation_add_tail` | `true` | Adds a bucket for the remaining probability mass instead of renormalizing the top-k. |
| `alpha` | `0.5` | KL direction: `0` forward KL(teacher ‖ student), `1` reverse KL(student ‖ teacher), in between a mixture. |
| `teacher_regularization` | `ema` | Teacher type; `ema` (EMA of the student) is the supported setting. |
| `teacher_update_rate` | `0.05` | EMA rate per step. |
| `is_clip` | `2.0` | Upper clip on the importance ratio between the current and rollout policies. |
| `include_environment_feedback` / `include_solution` / `include_answer` | `true` / `true` / `false` | Which privileged information is added to the teacher prompt. |
| `reprompt_template` / `feedback_template` / `solution_template` / `answer_template` | — | Templates for the teacher prompt. |
| `max_reprompt_len` | `10240` | Maximum teacher prompt length. |
| `token_mask_mode` | `none` | `none`, `entropy_diff_mask` (keep the lowest ΔH = H_student − H_teacher), or `reverse_entropy_diff_mask` (keep the highest ΔH). |
| `token_mask_top_pct` | `70.0` | The mask keeps (100 − `token_mask_top_pct`)% of each sequence's tokens: `70` keeps 30%. |

---

## Citation and attribution

This work builds on verl. If you use this code, please also cite:

> *HybridFlow: A Flexible and Efficient RLHF Framework.* Sheng et al., EuroSys 2025. [arXiv:2409.19256](https://arxiv.org/abs/2409.19256)

## License

Apache License 2.0, matching verl. Source files carry verl's Apache 2.0 header (© ByteDance Ltd. and/or its affiliates). See `LICENSE` and `Notice.txt`.
