"""
Preprocess allenai/reward-bench-2 to parquet for validation during OPSD-GenRM training.

RewardBench-2 is one-vs-many: each row has 1 chosen and 3 rejected responses
(4 chosen for the ``Ties`` subset, which we drop). We expand each row into
N-1 pairwise comparisons (chosen vs each rejected). Position is randomized
per pair.

Source-prompt grouping (all-correct reduction):
    Any data source whose preprocessor emits ``source_prompt_id`` in
    ``extra_info`` gets an additional reduction step: after per-uid metrics
    are computed, all uids sharing the same ``source_prompt_id`` are
    collapsed via ``np.min`` on every (variable, metric) pair. This
    implements the "all-correct" reduction — the source-prompt-level value
    is 1 only when every grouped sample is correct. Datasets without
    ``source_prompt_id`` skip this step and keep their per-uid metrics.

Metric semantics (matches RewardBench-2 Best-of-1):
  Each source row contributes ONE binary verdict: 1 if the judge correctly
  picks ``chosen`` over every rejected response, else 0 (one miss → 0).
  This certifies that the reward model would rank ``chosen`` as the top
  candidate in a Best-of-N selection over all 4 candidates. Each (chosen
  vs rejected_k) pair is emitted as ONE sample sharing one
  ``source_prompt_id``, so the all-correct reduction above produces the
  per-row verdict.

  With "rewardbench2" in ``MACRO_AVG_DATA_SOURCES``, the overall value is
  macro-by-subset of that Best-of-1 score.

Reference: https://huggingface.co/datasets/allenai/reward-bench-2
"""

import os
import random
import argparse

import datasets

from common import format_context, format_response
from prompt_templates import get_prompt_template, PROMPT_TEMPLATES


def extract_rewardbench2(example):
    """Expand one RB2 row into (chosen vs rejected_k) pairs with randomized
    position. Each pair has its own unique uid; all pairs from one row share
    ``source_prompt_id`` (set in extra_info) so metric_utils can group them
    for the all-correct reduction."""
    chosen = example["chosen"][0]
    pairs = []
    for rejected in example["rejected"]:
        if random.random() < 0.5:
            response1, response2, label = chosen, rejected, "A"
        else:
            response1, response2, label = rejected, chosen, "B"
        pairs.append(
            {
                "context": example["prompt"],
                "response1": response1,
                "response2": response2,
                "label": label,
                "domain": example["subset"],
                "source_id": example["id"],
            }
        )
    return pairs


def make_map_fn(include_format=False):
    def process_fn(example, idx):
        context = format_context(
            [{"role": "user", "content": example["context"]}],
            include_format=include_format,
        )
        resp_a = format_response(example["response1"], include_format=include_format)
        resp_b = format_response(example["response2"], include_format=include_format)
        user_msg = USER_MSG_TEMPLATE.format(
            context=context.rstrip(),
            response_a=resp_a,
            response_b=resp_b,
        )
        prompt = []
        if SYSTEM_MSG:
            prompt.append({"role": "system", "content": SYSTEM_MSG})
        prompt.append({"role": "user", "content": user_msg})
        return {
            "data_source": "rewardbench2",
            "prompt": prompt,
            "ability": "rm",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["label"],
            },
            "extra_info": {
                "domain": example["domain"],
                "source_prompt_id": f"rb2_{example['source_id']}",
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="datasets/rewardbench2",
                        help="Output dir for parquet files.")
    parser.add_argument("--include_format", action="store_true",
                        help="Wrap context messages and the two responses in chat-style tags.")
    parser.add_argument("--prompt_template", default="pair_rm",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help="Which prompt template variant to use (default: pair_rm)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for position randomization")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="If set, randomly subsample this many source "
                             "rows AFTER filtering Ties and BEFORE the "
                             "one-vs-many expansion (final parquet will "
                             "contain 3 * max_samples pairs).")
    args = parser.parse_args()

    random.seed(args.seed)

    _tmpl = get_prompt_template(args.prompt_template)
    SYSTEM_MSG = _tmpl.student_sys_msg
    USER_MSG_TEMPLATE = _tmpl.student_user_msg

    raw = datasets.load_dataset("allenai/reward-bench-2", split="test")
    raw = raw.filter(lambda ex: ex["subset"] != "Ties")

    if args.max_samples is not None and args.max_samples < len(raw):
        indices = random.sample(range(len(raw)), args.max_samples)
        raw = raw.select(sorted(indices))
        print(f"Subsampled RewardBench-2 source rows: {len(raw)}")

    all_pairs = []
    for example in raw:
        all_pairs.extend(extract_rewardbench2(example))
    pair_dataset = datasets.Dataset.from_list(all_pairs)

    test_dataset = pair_dataset.map(
        function=make_map_fn(include_format=args.include_format),
        with_indices=True,
        remove_columns=pair_dataset.column_names,
    )

    # Unique uid per pair. The all-correct reduction that groups pairs
    # sharing one source prompt is done in metric_utils.py, keyed off
    # extra_info["source_prompt_id"].
    def add_uid(example, idx):
        example["uid"] = f"rewardbench2_test_{idx}"
        return example

    test_dataset = test_dataset.map(function=add_uid, with_indices=True)

    sample = test_dataset[0]["prompt"]
    if len(sample) > 1:
        print("\n===== Sample System Message =====")
        print(sample[0]["content"])
        user_msg = sample[1]["content"]
    else:
        print("\n(No system message)")
        user_msg = sample[0]["content"]
    print("\n===== Sample User Message =====")
    print(user_msg[:500], "..." if len(user_msg) > 500 else "")

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"\nTest:  {len(test_dataset)} examples -> "
          f"{os.path.join(local_dir, 'test.parquet')}")
