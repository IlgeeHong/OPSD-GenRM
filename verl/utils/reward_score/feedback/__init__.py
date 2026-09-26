"""Feedback-aware reward scoring for self-distillation teacher batch construction.

These reward functions return dicts containing ``feedback`` and ``answer`` fields
(in addition to the usual ``score``), which are consumed by
:func:`verl.trainer.ppo.self_distillation.teacher_batch.build_teacher_batch`.
"""

from verl.utils.reward_score.feedback import binary_preference


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """Dispatch to the appropriate dataset-specific reward function.

    Any reward function that populates ``feedback`` and ``answer`` in the
    returned dict is compatible with the self-distillation teacher batch.
    """
    if data_source in (
        "helpsteer3",
        "helpsteer3_dedup",
        "rmbench",
        "rewardbench2",
    ):
        return binary_preference.compute_score(solution_str, ground_truth, extra_info)
    else:
        raise ValueError(
            f"Feedback reward function for data_source={data_source!r} not found. "
            "Supported: helpsteer3, helpsteer3_dedup, rmbench, rewardbench2"
        )
