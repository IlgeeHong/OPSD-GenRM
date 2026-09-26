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

"""Teacher batch construction for self-distillation.

The teacher sees the student's prompt enriched with privileged information
(annotator feedback, a successful peer rollout, and/or the ground-truth
answer) and scores the student's own responses under that prompt.
"""

import re
from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def _collect_feedback(
    include_environment_feedback: bool,
    reward_extra_infos_dict: Optional[dict[str, Any]],
    batch_size: int,
) -> list[Any]:
    """Collect environment feedback from reward_extra_infos_dict."""
    feedback_list: list[Any] = [None] * batch_size
    if include_environment_feedback and reward_extra_infos_dict is not None:
        raw_feedback = reward_extra_infos_dict.get("feedback", [])
        for i in range(min(len(raw_feedback), batch_size)):
            if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                feedback_list[i] = raw_feedback[i]
    return feedback_list


def _collect_solutions_by_uid(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    success_reward_threshold: float,
    min_response_len: int = 0,
) -> dict[Any, list[int]]:
    """Group successful rollout indices by uid."""
    seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
    uids = batch.non_tensor_batch["uid"]
    response_lengths = batch.batch["response_mask"].sum(dim=-1).cpu().numpy()
    success_by_uid: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        if seq_scores[idx] >= success_reward_threshold:
            if response_lengths[idx] < min_response_len:
                continue
            success_by_uid[uid].append(idx)
    return success_by_uid


def _remove_thinking_trace(text: str) -> str:
    """Remove <think>...</think> tags and their content from text."""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)


def _get_solution(
    idx: int,
    success_by_uid: dict[Any, list[int]],
    uids: list[Any],
    response_texts: list[str],
    dont_reprompt_on_self_success: bool = False,
    remove_thinking_from_demonstration: bool = False,
) -> Optional[str]:
    """Get a peer solution for sample idx from a successful rollout of the same uid."""
    uid = uids[idx]
    solution_idxs = success_by_uid[uid]
    if dont_reprompt_on_self_success:
        solution_idxs = [j for j in solution_idxs if j != idx]
    if len(solution_idxs) == 0:
        return None
    solution_idx = solution_idxs[0]
    solution_str = response_texts[solution_idx]
    if remove_thinking_from_demonstration:
        solution_str = _remove_thinking_trace(solution_str)
    return solution_str


def _compute_non_thinking_mask(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
) -> torch.Tensor:
    """Return a mask that zeros out content between <think> and </think>.

    The bracket tokens themselves are kept (mask=1) so the teacher sees
    an empty <think></think>. Only thinking content is masked (mask=0).
    """
    start_think_ids = tokenizer.encode("<think>", add_special_tokens=False)
    end_think_ids = tokenizer.encode("</think>", add_special_tokens=False)
    ns, ne = len(start_think_ids), len(end_think_ids)
    batch_size, response_length = responses.shape
    mask = response_mask.clone()
    for i in range(batch_size):
        row = responses[i].tolist()
        start_content = None
        for t in range(response_length - ns + 1):
            if row[t: t + ns] == start_think_ids:
                start_content = t + ns
                break
        end_content = None
        for t in range(response_length - ne + 1):
            if row[t: t + ne] == end_think_ids:
                end_content = t
                break
        if start_content is None and end_content is not None:
            start_content = 0
        if start_content is not None and end_content is not None and start_content < end_content:
            mask[i, start_content:end_content] = 0
    return mask


def _build_teacher_message(
    i: int,
    prompt_texts: list[str],
    batch: DataProto,
    solution_strs: list[Optional[str]],
    feedback_list: list[Any],
    answer_list: list[Any],
    self_distillation_cfg: Any,
    student_user_suffix: Optional[str],
) -> list[dict]:
    """Build the teacher message (system + user) for a single sample.

    All oracle slots (solution, feedback, answer) are spliced into
    ``self_distillation_cfg.reprompt_template`` via its ``{solution} {feedback}
    {answer}`` placeholders. The system message is taken verbatim from the
    original prompt; per-template teacher overrides are not supported.
    """
    system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]

    has_solution = solution_strs[i] is not None
    has_feedback = feedback_list[i] is not None
    use_answer = answer_list[i] is not None
    feedback_only_without_solution = getattr(
        self_distillation_cfg, "environment_feedback_only_without_solution", False
    )
    use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

    solution_section = ""
    if has_solution:
        solution_section = self_distillation_cfg.solution_template.format(
            successful_previous_attempt=solution_strs[i]
        )
    feedback_section = ""
    if use_feedback:
        feedback_section = self_distillation_cfg.feedback_template.format(
            feedback_raw=feedback_list[i]
        )
    answer_section = ""
    if use_answer:
        answer_section = self_distillation_cfg.answer_template.format(
            answer_raw=answer_list[i]
        )

    base_prompt = prompt_texts[i]
    if student_user_suffix is not None and base_prompt.endswith(student_user_suffix):
        base_prompt = base_prompt[:-len(student_user_suffix)]

    if use_feedback or has_solution or use_answer:
        reprompt_text = self_distillation_cfg.reprompt_template.format(
            prompt=base_prompt,
            solution=solution_section,
            feedback=feedback_section,
            answer=answer_section,
        )
    else:
        reprompt_text = base_prompt

    return system_messages + [{"role": "user", "content": reprompt_text}]


def build_teacher_batch(
    batch: DataProto,
    reward_tensor: torch.Tensor,
    reward_extra_infos_dict: Optional[dict[str, list]],
    tokenizer: Any,
    self_distillation_cfg: Any,
    apply_chat_template_kwargs: Optional[dict] = None,
) -> Optional[tuple[DataProto, dict[str, float]]]:
    """Build the teacher batch for self-distillation.

    Returns a DataProto with teacher_input_ids, teacher_attention_mask,
    teacher_position_ids, and self_distillation_mask, plus a metrics dict.
    Returns None when self-distillation is not configured.
    """
    if self_distillation_cfg is None:
        return None

    device = batch.batch["input_ids"].device
    response_mask = batch.batch["response_mask"]
    responses = batch.batch["responses"]
    response_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
    prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
    batch_size = batch.batch.batch_size[0]

    feedback_list = _collect_feedback(
        include_environment_feedback=self_distillation_cfg.include_environment_feedback,
        reward_extra_infos_dict=reward_extra_infos_dict,
        batch_size=batch_size,
    )

    include_answer = getattr(self_distillation_cfg, "include_answer", False)
    answer_list: list[Any] = [None] * batch_size
    if include_answer and reward_extra_infos_dict is not None:
        raw_answers = reward_extra_infos_dict.get("answer", [])
        for i in range(min(len(raw_answers), batch_size)):
            if raw_answers[i] and isinstance(raw_answers[i], str) and raw_answers[i].strip():
                answer_list[i] = raw_answers[i]

    include_solution = getattr(self_distillation_cfg, "include_solution", True)
    if include_solution:
        success_by_uid = _collect_solutions_by_uid(
            batch, reward_tensor,
            success_reward_threshold=self_distillation_cfg.success_reward_threshold,
            min_response_len=getattr(self_distillation_cfg, "min_solution_len", 0),
        )
        solution_strs = [
            _get_solution(
                i, success_by_uid, batch.non_tensor_batch["uid"], response_texts,
                self_distillation_cfg.dont_reprompt_on_self_success,
                getattr(self_distillation_cfg, "remove_thinking_from_demonstration", False),
            )
            for i in range(batch_size)
        ]
    else:
        success_by_uid = defaultdict(list)
        solution_strs = [None] * batch_size

    # The student's closing instruction is replaced by the one in reprompt_template.
    prompt_template_key = getattr(self_distillation_cfg, "prompt_template", None)
    student_user_suffix = None
    if prompt_template_key:
        from data.prompt_templates import get_prompt_template
        _tmpl = get_prompt_template(prompt_template_key)
        student_user_suffix = _tmpl.student_user_msg_suffix

    messages = [
        _build_teacher_message(
            i, prompt_texts, batch, solution_strs, feedback_list, answer_list,
            self_distillation_cfg, student_user_suffix,
        )
        for i in range(batch_size)
    ]

    enable_thinking = True
    if apply_chat_template_kwargs:
        enable_thinking = apply_chat_template_kwargs.get("enable_thinking", True)
    teacher_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        continue_final_message=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        max_length=self_distillation_cfg.max_reprompt_len,
        padding=True,
        truncation=True,
    )

    strip_thinking = getattr(self_distillation_cfg, "strip_thinking_from_sd_loss", False)
    if strip_thinking:
        non_thinking_mask = _compute_non_thinking_mask(responses, response_mask, tokenizer)
        teacher_response_attn = non_thinking_mask
    else:
        non_thinking_mask = None
        teacher_response_attn = response_mask

    teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
    teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), teacher_response_attn], dim=1)
    teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

    feedback_only_without_solution = getattr(self_distillation_cfg, "environment_feedback_only_without_solution", False)
    feedback_used = [
        feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
        for i in range(batch_size)
    ]

    has_teacher_signal = [
        solution_strs[i] is not None or feedback_used[i] or answer_list[i] is not None
        for i in range(batch_size)
    ]

    distill_only_on_success = getattr(self_distillation_cfg, "distill_only_on_success", False)
    if distill_only_on_success:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        threshold = self_distillation_cfg.success_reward_threshold
        has_teacher_signal = [
            has_teacher_signal[i] and seq_scores[i] >= threshold
            for i in range(batch_size)
        ]

    self_distillation_mask = torch.tensor(has_teacher_signal, dtype=torch.float32, device=device)

    uids = set(batch.non_tensor_batch["uid"])
    num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
    num_with_feedback_used = sum(1 for f in feedback_used if f)
    num_with_solution = sum(1 for s in solution_strs if s is not None)
    sd_metrics = {
        "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
        "self_distillation/success_sample_fraction": num_with_solution / batch_size,
        "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
        "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
        "self_distillation/answer_used_fraction": sum(1 for a in answer_list if a is not None) / batch_size,
        "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
    }

    # Per-token mask when thinking content is excluded from the loss.
    if non_thinking_mask is not None:
        self_distillation_mask = self_distillation_mask.unsqueeze(1) * non_thinking_mask

    result_tensors = {
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attention_mask,
        "teacher_position_ids": teacher_position_ids,
        "self_distillation_mask": self_distillation_mask,
    }
    return DataProto.from_dict(tensors=result_tensors), sd_metrics
