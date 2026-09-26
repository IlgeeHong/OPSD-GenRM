"""Binary preference reward scoring with annotator feedback extraction.

Used for HelpSteer3, RM-Bench, and RewardBench 2, where the model
must choose between two assistant responses (A or B) and annotator feedback is
available for self-distillation teacher prompts.
"""

import re


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text using <verdict> tags."""
    answer = text.split("<verdict>")[-1]
    answer = answer.split("</verdict>")[0]
    return answer.strip()


def is_correct_format(text: str) -> bool:
    """Check if the text ends with <verdict>(A|B)</verdict>."""
    pattern = r"<verdict>\s*(A|B)\s*</verdict>$"
    return re.search(pattern, text) is not None


def compute_score(solution: str, ground_truth: str, extra_info: dict = None) -> dict:
    """Score a binary preference response and extract annotator feedback.

    Returns a dict with keys: score, acc, pred, incorrect_format, feedback,
    answer, domain.  The ``feedback`` and ``answer`` fields are consumed by
    :func:`verl.trainer.ppo.self_distillation.teacher_batch.build_teacher_batch`.
    """
    pred = extract_xml_answer(solution)

    reward = float(pred == ground_truth)
    correct_format = is_correct_format(solution)

    feedback_mode = extra_info.get("feedback_mode") if extra_info else None
    if feedback_mode == "rubric":
        # Rubrics are attached to the dataset during preprocessing; use them verbatim.
        feedback = (extra_info or {}).get("rubric", "") or ""
    elif feedback_mode:
        feedback = _extract_best_annotator(extra_info, mode=feedback_mode)
    else:
        feedback = ""

    return {
        "score": reward,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 0 if correct_format else 1,
        "feedback": feedback,
        "answer": ground_truth,
        "domain": (extra_info.get("domain") or "") if extra_info else "",
        "source_prompt_id": (extra_info.get("source_prompt_id") or "") if extra_info else "",
    }


def _extract_best_annotator(extra_info: dict, mode: str = "reasoning") -> str:
    """Select a single annotator whose score matches overall_preference.

    Selection priority:
      1. Exact match: individual score == overall_preference
      2. Direction match: same sign as overall_preference
      3. First annotator with non-empty content

    Args:
        extra_info: Dict with individual_preference, overall_preference, etc.
        mode: "reasoning" uses the reasoning field (one text covering both responses).
              "feedback" uses feedback1 + feedback2 fields (separate per-response evaluations).
    """
    if not extra_info:
        return ""
    individual_prefs = extra_info.get("individual_preference", [])
    if len(individual_prefs) == 0:
        return ""
    overall_pref = extra_info.get("overall_preference")
    position_flipped = extra_info.get("position_flipped", False)

    if overall_pref is None:
        for pref in individual_prefs:
            text = _format_single_annotator(pref, position_flipped, mode)
            if text:
                return text
        return ""

    # 1. Try exact match
    for pref in individual_prefs:
        score = pref.get("score", 0)
        if score == overall_pref:
            text = _format_single_annotator(pref, position_flipped, mode)
            if text:
                return text

    # 2. Try direction match (same sign)
    for pref in individual_prefs:
        score = pref.get("score", 0)
        if score != 0 and (score > 0) == (overall_pref > 0):
            text = _format_single_annotator(pref, position_flipped, mode)
            if text:
                return text

    # 3. Fallback: first with content
    for pref in individual_prefs:
        text = _format_single_annotator(pref, position_flipped, mode)
        if text:
            return text

    return ""


def _format_single_annotator(pref: dict, position_flipped: bool, mode: str) -> str:
    """Format a single annotator's feedback based on mode."""
    if mode == "reasoning":
        reasoning = pref.get("reasoning", "").strip()
        if reasoning:
            return _replace_response_refs(reasoning, position_flipped)
        return ""
    elif mode == "feedback":
        fb1 = pref.get("feedback1", "").strip()
        fb2 = pref.get("feedback2", "").strip()
        if not fb1 and not fb2:
            return ""
        if position_flipped:
            fb_a, fb_b = fb2, fb1
        else:
            fb_a, fb_b = fb1, fb2
        lines = []
        if fb_a:
            lines.append(f"Feedback on Assistant A's response: {fb_a}")
        if fb_b:
            lines.append(f"Feedback on Assistant B's response: {fb_b}")
        return "\n\n".join(lines)
    return ""


def _replace_response_refs(text: str, position_flipped: bool) -> str:
    """Replace @Response 1/@Response 2 with Assistant A/B labels.

    When position_flipped=True, response1 was placed as B and response2 as A
    in the user message, so the mapping is reversed.
    """
    if position_flipped:
        text = text.replace("@Response 1", "Assistant B's response")
        text = text.replace("@Response 2", "Assistant A's response")
    else:
        text = text.replace("@Response 1", "Assistant A's response")
        text = text.replace("@Response 2", "Assistant B's response")
    return text
