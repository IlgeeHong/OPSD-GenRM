# OPSD-GenRM

📄 **Paper:** [Training LLM Judges from Language Feedback via Position-Selective Self-Distillation](https://arxiv.org/abs/2609.38792v1)

OPSD-GenRM implements *Training LLM Judges from Language Feedback via Position-Selective Self-Distillation*. The paper studies how to train a generative reward model (a pairwise LLM judge) from language feedback. The GRPO baseline uses the preference label to reward the final verdict; self-distillation uses the annotator's language feedback to provide token-level supervision, including at criterion choice positions. Compared with unmasked self-distillation, the position-selective variant improves out-of-distribution generalization. The paper also reports that OPSD-GenRM significantly outperforms the GRPO baseline on subjective tasks. This implementation builds on [verl](https://github.com/volcengine/verl) (`release/v0.7.1`) and [SDPO](https://github.com/lasgroup/SDPO).

For each preference pair, OPSD-GenRM first identifies evaluation criteria (or rubric) that matter for the given user prompt, then compares the two responses along those criteria and outputs a final A/B verdict.

---

## Trained models

The trained models are available at [huggingface.co/opsd-genrm/models](https://huggingface.co/opsd-genrm/models).

---

## Installation

Follow [verl's v0.7.1 installation guide](https://github.com/verl-project/verl/blob/release/v0.7.1/docs/start/install.rst) to select and run a Docker image compatible with your GPU. These recipes use FSDP for training and vLLM for rollouts, so choose an image with vLLM support. Inside that environment, install this repository:

```bash
git clone https://github.com/IlgeeHong/OPSD-GenRM.git
cd OPSD-GenRM
pip install --no-deps -e .
```

The environment needs `vllm`, `flash-attn`, and `triton`; the fused top-k kernel requires `triton >= 2.3`.

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

The launchers run these scripts automatically. `--feedback_mode` selects the teacher's privileged information: `reasoning` (annotator rationale) or `rubric`. `--prompt_template` selects the judge instruction: `pair_rm` asks for task-specific quality dimensions before comparison; `pair_rm_rubric` asks for a ranked task-specific rubric before comparison. The teacher prompt uses the same instruction as the student, so the two differ only by the feedback. Each template's data is written to `datasets/<template>/`.

---

## Reproducing the experiments

Each launcher preprocesses the data (into `datasets/`), trains, and converts every saved checkpoint to Hugging Face format (under `checkpoints/<experiment>/hf_step_<N>`).

| Launcher | Method | Teacher feedback | Token mask |
|---|---|---|---|
| `run_drgrpo.sh` | Dr. GRPO baseline | — | — |
| `run_rationale_sd.sh` | Self-distillation | annotator rationale | none |
| `run_rationale_sd_mask70.sh` | Position-selective self-distillation | annotator rationale | `entropy_diff_mask`, bottom 30% of ΔH |
| `run_rubric_sd.sh` | Self-distillation | task rubric | none |
| `run_rubric_sd_mask70.sh` | Position-selective self-distillation | task rubric | `entropy_diff_mask`, bottom 30% of ΔH |

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

### Self-distillation configuration

The settings below are under `actor_rollout_ref.actor.self_distillation` unless a full path is shown. The launchers set the paper's choices explicitly; other defaults are in `verl/trainer/config/actor/actor.yaml`.

| Setting | Meaning in these recipes |
|---|---|
| `enable=true`, `actor_rollout_ref.actor.policy_loss.loss_mode=full_logit_kl` | Build a feedback-conditioned teacher and train with the token-distribution KL loss. |
| `prompt_template` | Select the pairwise judge instruction shared by student and teacher: `pair_rm` for rationale runs, `pair_rm_rubric` for rubric runs. |
| `reprompt_template`, `feedback_template` | Format the teacher prompt and insert the rationale or rubric. The rationale and rubric Hydra configs supply their respective templates. |
| `include_environment_feedback=true`, `include_solution=false`, `include_answer=false` | Give the teacher the language feedback, without adding another rollout or a reference answer. |
| `max_reprompt_len` | Maximum token length of the teacher prompt before scoring the student's response: 4608 for rationale runs, 5120 for rubric runs. |
| `teacher_regularization=ema`, `teacher_update_rate=0.01` | Use an EMA copy of the student as teacher and update it with rate 0.01. |
| `distillation_topk=100`, `distillation_add_tail=true` | Compute the KL approximation over the teacher's top 100 tokens plus one bucket containing all remaining probability mass. |
| `alpha=1` | Use reverse KL, from student to teacher, rather than forward KL. |
| `token_mask_mode`, `token_mask_top_pct` | `none` uses all valid response tokens. `entropy_diff_mask` with `token_mask_top_pct=70` removes the 70% with the largest ΔH in each response, retaining the lowest 30%. The percentage is ignored when the mode is `none`. |
| `is_clip=2.0` | Clip the importance ratio between the current student and the policy that generated the rollout when weighting the KL loss. |

### New modules

| Path | Purpose |
|---|---|
| `verl/trainer/ppo/self_distillation/teacher_batch.py` | Builds the teacher prompts (student prompt + feedback, optionally a successful peer rollout or the answer) and the per-sample self-distillation mask. |
| `verl/trainer/ppo/self_distillation/loss.py` | `full_logit_kl` policy loss: top-k KL between student and teacher with a tail bucket, importance-ratio clipping, and the entropy-based token masks. |
| `verl/trainer/ppo/self_distillation/teacher_model.py` | EMA teacher update. |
| `verl/trainer/ppo/self_distillation/reward.py` | `sd` advantage estimator, used only with `policy_loss.loss_mode=vanilla`. |
| `verl/utils/kernel/topk_logprobs.py` | Fused Triton kernel for top-k log-probabilities without materializing the full log-softmax. |
| `verl/utils/reward_score/feedback/` | Pairwise verdict reward (`<verdict>A\|B</verdict>`) that also returns the annotator feedback or rubric for the teacher. |
| `verl/utils/reward_score/feedback/prompt_templates.py` | Pairwise judge prompt templates (`pair_rm`, `pair_rm_rubric`), shared by the preprocessing scripts, the per-epoch A/B swap, and the teacher prompt builder. |
| `verl/trainer/ppo/token_mask.py` | Teacher-free entropy token mask for GRPO-style losses. |
| `verl/trainer/config/opsd_genrm.yaml` | Hydra config with the teacher prompt template for the rationale launchers. |
| `verl/trainer/config/opsd_genrm_rubric.yaml` | Same for the rubric launchers, matching the `pair_rm_rubric` student instruction. |
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

## Citation and attribution

This code builds on [verl](https://github.com/volcengine/verl) and [SDPO](https://github.com/lasgroup/SDPO).

> *HybridFlow: A Flexible and Efficient RLHF Framework.* Sheng et al., EuroSys 2025. [arXiv:2409.19256](https://arxiv.org/abs/2409.19256)

> *Reinforcement Learning via Self-Distillation.* Hübotter et al., 2026. [arXiv:2601.20802](https://arxiv.org/abs/2601.20802)

## License

Apache License 2.0, matching verl and SDPO. Source files carry verl's Apache 2.0 header (© ByteDance Ltd. and/or its affiliates). See `LICENSE` and `Notice.txt`.
