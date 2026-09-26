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

"""
Prompt templates for OPSD-GenRM pairwise judging.

Each ``PromptTemplate`` defines:
  - ``student_user_msg``: full user message (context + responses + closing
    instruction). Used by the data preprocessor and by the swap-order
    re-rendering path in ``rl_dataset.py``.
  - ``student_sys_msg``: optional system message inserted before the user
    message. None means no system role is emitted.
  - ``student_user_msg_suffix``: the closing instruction tail of
    ``student_user_msg``. The self-distillation teacher-batch builder strips this from the
    base prompt before splicing in oracle slots via the
    ``self_distillation.reprompt_template`` config; the ``{ground_truth}``
    label, when ``include_answer`` is enabled, flows through the
    ``{answer}`` slot of that same template.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# User message: context + two responses
# ---------------------------------------------------------------------------

PAIR_RM_USER_MSG = (
    "You are an impartial judge tasked with determining which of two assistant responses "
    "is better for the given context.\n\n"
    "Below is a context (a user query or a conversation between the user and an assistant) "
    "and two assistant responses to that context.\n\n"
    "[Start of Context]\n"
    "{context}\n"
    "[End of Context]\n\n"
    "[Start of Assistant A's Response]\n"
    "{response_a}\n"
    "[End of Assistant A's Response]\n\n"
    "[Start of Assistant B's Response]\n"
    "{response_b}\n"
    "[End of Assistant B's Response]"
)


# ---------------------------------------------------------------------------
# Closing instruction
# ---------------------------------------------------------------------------

# Pairwise reward model: identify task-specific dimensions, compare step by step,
# verify correctness when relevant, emit an A/B verdict.
PAIR_RM_STUDENT_SUFFIX = (
    "\n\nIdentify the quality dimensions that matter most for this specific task, then "
    "evaluate and compare the two assistant responses step by step across those dimensions. "
    "When correctness matters, solve the problem yourself and check each response for any errors. "
    "After your analysis, determine which response is better overall and provide your final verdict "
    "(A or B only) in <verdict>...</verdict>."
)

# Rubric variant: derive a task-specific rubric first, then compare against it.
# Must match the closing instruction in verl/trainer/config/opsd_genrm_rubric.yaml,
# so that student and teacher prompts differ only by the rubric feedback.
PAIR_RM_RUBRIC_STUDENT_SUFFIX = (
    "\n\nIdentify the rubric that matters most for this specific task: the hard requirements the "
    "response must satisfy, ranked by importance, and the discriminative criteria that most decisively "
    "separate a better response from a worse one, ranked most decisive first \u2014 stating for each what "
    "makes a response better versus worse. Then evaluate and compare the two assistant responses step by "
    "step against that rubric. When correctness matters, solve the problem yourself and check each response "
    "for any errors. After your analysis, determine which response is better overall and provide your final "
    "verdict (A or B only) in <verdict>...</verdict>."
)

# ---------------------------------------------------------------------------
# PromptTemplate dataclass and lookup
# ---------------------------------------------------------------------------

@dataclass
class PromptTemplate:
    """Groups all prompt components for a single evaluation variant."""

    student_user_msg: str
    student_sys_msg: str | None = None
    student_user_msg_suffix: str | None = None


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "pair_rm": PromptTemplate(
        student_user_msg=PAIR_RM_USER_MSG + PAIR_RM_STUDENT_SUFFIX,
        student_user_msg_suffix=PAIR_RM_STUDENT_SUFFIX,
    ),
    "pair_rm_rubric": PromptTemplate(
        student_user_msg=PAIR_RM_USER_MSG + PAIR_RM_RUBRIC_STUDENT_SUFFIX,
        student_user_msg_suffix=PAIR_RM_RUBRIC_STUDENT_SUFFIX,
    ),
}


def get_prompt_template(name: str) -> PromptTemplate:
    """Return the PromptTemplate for the given variant name."""
    if name not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template variant '{name}'. Choose from: {list(PROMPT_TEMPLATES.keys())}")
    return PROMPT_TEMPLATES[name]
