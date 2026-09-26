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

"""Entropy-based token masking for teacher-free losses (e.g. GRPO).

Keeps the highest-entropy (100 - top_pct)% of response tokens in each sequence,
following "Beyond the 80/20 Rule" (arXiv:2506.01939). The self-distillation
counterpart is ``verl.trainer.ppo.self_distillation.loss._apply_entropy_mask``.
"""

import warnings
from typing import Optional

import torch


def apply_high_entropy_mask(
    loss_mask: torch.Tensor,
    policy_entropy: Optional[torch.Tensor],
    top_pct: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mask out low-entropy tokens per-sequence; keep the high-entropy tail.

    Args:
        loss_mask: (batch, seq_len) existing response mask (1 for positions
            contributing to the loss, 0 otherwise).
        policy_entropy: (batch, seq_len) per-token policy entropy.
        top_pct: percentile cutoff. E.g. top_pct=70 keeps tokens whose entropy
            is >= the 70th percentile inside each sequence, i.e. the top 30%.

    Returns:
        (mask, metrics) where mask has the same shape as loss_mask.
    """
    metrics: dict[str, float] = {}

    if policy_entropy is None:
        warnings.warn(
            "apply_high_entropy_mask called with policy_entropy=None. "
            "Set actor.calculate_entropy=True. Falling back to uniform weighting.",
            stacklevel=2,
        )
        return loss_mask, metrics

    seq_lens = loss_mask.sum(dim=-1).long()
    score_for_sort = policy_entropy.masked_fill(~loss_mask.bool(), float("-inf"))
    sorted_vals, _ = score_for_sort.sort(dim=-1)

    thresholds = torch.full_like(seq_lens, 0, dtype=policy_entropy.dtype).fill_(float("-inf"))
    for b in range(policy_entropy.shape[0]):
        n = seq_lens[b].item()
        if n > 0:
            idx = policy_entropy.shape[1] - n + int(top_pct / 100.0 * n)
            idx = min(idx, policy_entropy.shape[1] - 1)
            thresholds[b] = sorted_vals[b, idx]

    mask = (policy_entropy >= thresholds.unsqueeze(1)).float() * loss_mask
    metrics["actor/entropy_mask_frac"] = mask.sum().item() / loss_mask.sum().clamp(min=1).item()
    return mask, metrics
