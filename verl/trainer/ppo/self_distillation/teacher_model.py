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

from types import SimpleNamespace

import torch
from torch import nn


class TrustRegionTeacher(nn.Module):
    """Teacher model that interpolates between reference and student logits.

    Used for trust-region regularization in self-distillation. The teacher's logits are
    a linear interpolation: (1 - mix_coef) * ref_logits + mix_coef * student_logits.
    """

    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


def update_ema_teacher(teacher_module: nn.Module, student_module: nn.Module, update_rate: float) -> None:
    """Update teacher parameters via exponential moving average of student parameters.

    teacher_param = (1 - update_rate) * teacher_param + update_rate * student_param
    """
    with torch.no_grad():
        for teacher_param, student_param in zip(teacher_module.parameters(), student_module.parameters()):
            student_data = student_param.data.to(device=teacher_param.device)
            teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
