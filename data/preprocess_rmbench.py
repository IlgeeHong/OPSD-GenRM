"""
Preprocess THU-KEG/RM-Bench dataset to parquet for validation during OPSD-GenRM training.

RM-Bench contains 3 chosen and 3 rejected responses per sample at increasing
style-bias levels. We create all 3x3 combinations (9 pairs per sample) with
difficulty determined by the chosen/rejected index relationship:
  - i == j  -> "normal"
  - i < j   -> "hard"   (chosen less stylish, rejected more stylish)
  - i > j   -> "easy"   (chosen more stylish, rejected less stylish)

Each pair is emitted once with randomized A/B position and a unique uid.
With "rmbench" in ``MACRO_AVG_DATA_SOURCES``, the overall metric is
macro-averaged over the ``domain`` field; ``difficulty`` is preserved in
extra_info for downstream slicing.

Reference: https://huggingface.co/datasets/THU-KEG/RM-Bench
"""

import os
import random
import argparse

import datasets

from verl.utils.reward_score.feedback.prompt_templates import PROMPT_TEMPLATES, get_prompt_template

from common import format_context, format_response


def extract_rm_bench(example):
    """Extract all 3x3 chosen/rejected combinations from a single RM-Bench example."""
    combinations = []
    for i, chosen_resp in enumerate(example["chosen"]):
        for j, rejected_resp in enumerate(example["rejected"]):
            if random.random() < 0.5:
                response1, response2, label = chosen_resp, rejected_resp, "A"
            else:
                response1, response2, label = rejected_resp, chosen_resp, "B"
            combinations.append(
                {
                    "context": example["prompt"],
                    "response1": response1,
                    "response2": response2,
                    "label": label,
                    "domain": "safety" if "safety" in example["domain"] else example["domain"],
                    "difficulty": "normal" if i == j else ("hard" if i < j else "easy"),
                }
            )
    return combinations


def make_map_fn(include_format=False):
    def process_fn(example, idx):
        context = format_context([{"role": "user", "content": example["context"]}], include_format=include_format)
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
            "data_source": "rmbench",
            "prompt": prompt,
            "ability": "rm",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["label"],
            },
            "extra_info": {
                "domain": example["domain"],
                "difficulty": example["difficulty"],
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="datasets/rmbench",
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
                             "rows BEFORE the 3x3 expansion (final parquet "
                             "will contain 9 * max_samples pairs).")
    args = parser.parse_args()

    random.seed(args.seed)

    _tmpl = get_prompt_template(args.prompt_template)
    SYSTEM_MSG = _tmpl.student_sys_msg
    USER_MSG_TEMPLATE = _tmpl.student_user_msg

    dataset = datasets.load_dataset("THU-KEG/RM-Bench")
    raw = dataset["train"]

    if args.max_samples is not None and args.max_samples < len(raw):
        indices = random.sample(range(len(raw)), args.max_samples)
        raw = raw.select(sorted(indices))
        print(f"Subsampled RM-Bench source rows: {len(raw)}")

    # Expand each example into all 3x3 chosen/rejected combinations
    all_combos = []
    for example in raw:
        all_combos.extend(extract_rm_bench(example))
    combo_dataset = datasets.Dataset.from_list(all_combos)

    test_dataset = combo_dataset.map(
        function=make_map_fn(include_format=args.include_format), with_indices=True,
        remove_columns=combo_dataset.column_names,
    )

    def add_uid(example, idx):
        example["uid"] = f"rmbench_test_{idx}"
        return example

    test_dataset = test_dataset.map(function=add_uid, with_indices=True)

    # Print one sample for verification
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

    print(f"\nTest:  {len(test_dataset)} examples -> {os.path.join(local_dir, 'test.parquet')}")
