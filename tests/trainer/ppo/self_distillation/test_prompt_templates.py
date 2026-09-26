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
"""Student and teacher prompts must differ only by the privileged feedback."""

from pathlib import Path

import pytest
import yaml

from verl.utils.reward_score.feedback.prompt_templates import get_prompt_template

CONFIG_DIR = Path(__file__).resolve().parents[4] / "verl" / "trainer" / "config"


@pytest.mark.parametrize(
    "template_name, config_name",
    [("pair_rm", "opsd_genrm.yaml"), ("pair_rm_rubric", "opsd_genrm_rubric.yaml")],
)
def test_teacher_prompt_is_student_prompt_plus_feedback(template_name, config_name):
    template = get_prompt_template(template_name)
    sd_cfg = yaml.safe_load((CONFIG_DIR / config_name).read_text())["actor_rollout_ref"]["actor"]["self_distillation"]

    student_prompt = template.student_user_msg.format(context="CONTEXT", response_a="A", response_b="B")
    assert student_prompt.endswith(template.student_user_msg_suffix)

    # Mirrors teacher_batch._build_teacher_message with include_solution/include_answer disabled.
    base_prompt = student_prompt[: -len(template.student_user_msg_suffix)]
    feedback = sd_cfg["feedback_template"].format(feedback_raw="FEEDBACK")
    teacher_prompt = sd_cfg["reprompt_template"].format(prompt=base_prompt, solution="", feedback=feedback, answer="")

    assert teacher_prompt == base_prompt + "\n\nFEEDBACK" + template.student_user_msg_suffix
