# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Top-k KL distillation loss for self-distillation.

Registered as ``full_logit_kl`` (and the alias ``sd``) through
``register_policy_loss`` so it plugs into the standard ``update_policy`` path.
"""

import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss


def _compute_kl_per_token(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    config: Any,
) -> torch.Tensor:
    """Per-token KL divergence between student and teacher over the top-k support.

    The top-k log-probs are either extended with a tail bucket holding the
    remaining probability mass (``distillation_add_tail``) or renormalized.
    ``alpha`` selects forward KL (0), reverse KL (1), or a mixture in between.

    Args:
        student_topk_log_probs: (batch, seq_len, k) student log-probs at the top-k indices.
        teacher_topk_log_probs: (batch, seq_len, k) teacher log-probs at the same indices.
        config: ``SelfDistillationConfig``.

    Returns:
        (batch, seq_len) KL per token.
    """
    if student_topk_log_probs is None or teacher_topk_log_probs is None:
        raise ValueError("full_logit_kl loss requires student_topk_log_probs and teacher_topk_log_probs.")

    if config.distillation_add_tail:
        def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
            log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_s = torch.clamp(log_s, max=-1e-7)
            tail_log = torch.log(-torch.expm1(log_s))
            return torch.cat([log_probs, tail_log], dim=-1)

        student_lp = add_tail(student_topk_log_probs)
        teacher_lp = add_tail(teacher_topk_log_probs)
    else:
        def renorm(logp: torch.Tensor) -> torch.Tensor:
            return logp - torch.logsumexp(logp, dim=-1, keepdim=True)

        student_lp = renorm(student_topk_log_probs)
        teacher_lp = renorm(teacher_topk_log_probs)

    alpha = config.alpha
    if alpha == 0.0:
        # Forward KL: D_KL(teacher || student)
        kl = F.kl_div(student_lp, teacher_lp, reduction="none", log_target=True)
    elif alpha == 1.0:
        # Reverse KL: D_KL(student || teacher)
        kl = F.kl_div(teacher_lp, student_lp, reduction="none", log_target=True)
    else:
        # Mixture KL
        alpha_t = torch.tensor(alpha, dtype=student_lp.dtype, device=student_lp.device)
        mixture = torch.logsumexp(
            torch.stack([student_lp + torch.log(1 - alpha_t), teacher_lp + torch.log(alpha_t)]),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture, teacher_lp, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture, student_lp, reduction="none", log_target=True)
        kl = torch.lerp(kl_student, kl_teacher, alpha)

    return kl.sum(-1)  # (batch, seq_len)


def _apply_entropy_mask(
    per_token_loss: torch.Tensor,
    loss_mask: torch.Tensor,
    config: Any,
    student_entropy: Optional[torch.Tensor],
    teacher_entropy: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Restrict the loss to a per-sequence subset of tokens ranked by an entropy score.

    Every mode keeps (100 - ``token_mask_top_pct``)% of the valid tokens in each
    sequence; ``token_mask_mode`` selects the score and which end is kept. With the
    entropy difference defined as dH = H_student - H_teacher:

    - ``entropy_diff_mask``: keep the tokens with the lowest dH.
    - ``reverse_entropy_diff_mask``: keep the tokens with the highest dH.
    - ``abs_entropy_diff_mask``: keep the tokens with the highest |dH|.
    - ``ht_mask`` / ``hs_mask``: keep the tokens with the highest teacher / student entropy.

    Returns the new loss mask and the kept-token fraction as a metric.
    """
    metrics = {}
    entropy_mode = getattr(config, "token_mask_mode", "none")

    if entropy_mode == "none":
        return loss_mask, metrics
    supported = ("entropy_diff_mask", "reverse_entropy_diff_mask", "abs_entropy_diff_mask", "ht_mask", "hs_mask")
    if entropy_mode not in supported:
        raise ValueError(f"Unsupported self_distillation.token_mask_mode={entropy_mode!r}. Choose from: none, {', '.join(supported)}.")
    needs_teacher = entropy_mode != "hs_mask"
    needs_student = entropy_mode != "ht_mask"
    if (needs_teacher and teacher_entropy is None) or (needs_student and student_entropy is None):
        warnings.warn(
            f"token_mask_mode={entropy_mode!r} but the required entropy is None. "
            "Set actor.calculate_entropy=True. Falling back to uniform weighting.",
            stacklevel=2,
        )
        return loss_mask, metrics

    percentile = getattr(config, "token_mask_top_pct", 70.0)
    seq_lens = loss_mask.sum(dim=-1).long()

    def _percentile_mask(score: torch.Tensor, metric_name: str, keep: str = "highest") -> torch.Tensor:
        """Keep the (100 - percentile)% highest- or lowest-scoring valid tokens per sequence."""
        if keep == "lowest":
            score = -score
        score_for_sort = score.masked_fill(~loss_mask.bool(), float("-inf"))
        sorted_vals, _ = score_for_sort.sort(dim=-1)
        thresholds = torch.full_like(seq_lens, float("-inf"), dtype=score.dtype)
        for b in range(score.shape[0]):
            n = seq_lens[b].item()
            if n > 0:
                idx = score.shape[1] - n + int(percentile / 100.0 * n)
                idx = min(idx, score.shape[1] - 1)
                thresholds[b] = sorted_vals[b, idx]
        mask = (score >= thresholds.unsqueeze(1)).float() * loss_mask
        metrics[metric_name] = mask.sum().item() / loss_mask.sum().clamp(min=1).item()
        return mask

    if entropy_mode in ("entropy_diff_mask", "reverse_entropy_diff_mask", "abs_entropy_diff_mask"):
        entropy_diff = student_entropy - teacher_entropy
    if entropy_mode == "entropy_diff_mask":
        return _percentile_mask(entropy_diff, "actor/entropy_mask_frac", keep="lowest"), metrics
    if entropy_mode == "reverse_entropy_diff_mask":
        return _percentile_mask(entropy_diff, "actor/reverse_entropy_mask_frac", keep="highest"), metrics
    if entropy_mode == "abs_entropy_diff_mask":
        return _percentile_mask(entropy_diff.abs(), "actor/abs_entropy_mask_frac"), metrics
    if entropy_mode == "ht_mask":
        mask = _percentile_mask(teacher_entropy, "actor/ht_mask_frac")
        return mask, metrics
    # hs_mask
    mask = _percentile_mask(student_entropy, "actor/hs_mask_frac")
    return mask, metrics


@register_policy_loss("full_logit_kl")
def compute_policy_loss_full_logit_kl(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_log_probs: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    student_entropy: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Top-k KL distillation loss between the student and the self-teacher.

    Uses the standard policy-loss interface plus teacher outputs as keyword
    arguments. Tokens are weighted by the clipped importance ratio and by
    ``self_distillation_mask``, which drops samples without a teacher signal.
    Entropy-based token masking is applied by the caller via ``response_mask``.
    """
    sd_cfg = getattr(config, "self_distillation", None)
    if sd_cfg is None:
        raise ValueError("full_logit_kl loss requires self_distillation config in actor config.")

    metrics = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        if self_distillation_mask.dim() == 1:
            loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
        else:
            loss_mask = loss_mask * self_distillation_mask

    per_token_loss = _compute_kl_per_token(
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        config=sd_cfg,
    )

    # Importance ratio between the current and rollout-time policy, clipped from above.
    is_clip = sd_cfg.is_clip
    if is_clip is not None:
        if old_log_prob is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        negative_approx_kl = (log_prob - old_log_prob).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )

    metrics["self_distillation/empty_target_batch"] = (
        self_distillation_mask.sum().item() == 0
        if self_distillation_mask is not None else False
    )

    return loss, metrics


@register_policy_loss("sd")
def compute_sd_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Any = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Alias of ``full_logit_kl``."""
    return compute_policy_loss_full_logit_kl(
        old_log_prob=old_log_prob, log_prob=log_prob,
        advantages=advantages, response_mask=response_mask,
        loss_agg_mode=loss_agg_mode, config=config,
        rollout_is_weights=rollout_is_weights, **kwargs,
    )
