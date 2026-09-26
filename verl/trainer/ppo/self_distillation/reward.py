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

"""Advantage estimator for self-distillation with the vanilla policy-gradient loss.

When ``policy_loss.loss_mode=vanilla``, the trainer sets ``token_level_rewards``
to the sampled-token log-ratio ``log pi_teacher(y_t) - log pi_student(y_t)`` and
this estimator passes it through as the per-token advantage. The ``full_logit_kl``
loss does not use an advantage estimator.
"""

from typing import Any, Optional

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("sd")
def compute_sd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Optional[np.ndarray] = None,
    config: Optional[Any] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the per-token teacher-student log-ratio directly as the advantage.

    Args:
        token_level_rewards: (batch, response_length) per-token log-ratios.
        response_mask: (batch, response_length) response token mask.
        index: Unused; kept for the estimator interface.
        config: Unused; kept for the estimator interface.

    Returns:
        (advantages, returns), both of shape (batch, response_length).
    """
    advantages = token_level_rewards * response_mask
    return advantages, advantages
