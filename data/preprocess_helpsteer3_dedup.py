"""Preprocess a deduplicated HelpSteer3 variant to parquet for OPSD-GenRM training.

Source HF dataset is a pre-cleaned variant of nvidia/HelpSteer3 (preference config):
  - multilingual and tie rows already removed
  - within-split dedup by content hash (keep-first)
  - cross-split dedup: val rows that collide with train are dropped
  - `rubric` column already attached (LLM-generated, 100% coverage on train)
"""

import os
import random
import argparse

import datasets

from verl.utils.reward_score.feedback.prompt_templates import PROMPT_TEMPLATES, get_prompt_template

from common import format_context, format_response


def make_map_fn(split, include_format=False):
    def process_fn(example, idx):
        context = format_context(example["context"], include_format=include_format)
        resp_a = format_response(example["response1"], include_format=include_format)
        resp_b = format_response(example["response2"], include_format=include_format)
        solution = "A" if example["overall_preference"] < 0 else "B"
        position_flipped = random.random() < 0.5
        if position_flipped:
            resp_a, resp_b = resp_b, resp_a
            solution = "B" if solution == "A" else "A"
        user_msg = USER_MSG_TEMPLATE.format(
            context=context.rstrip(),
            response_a=resp_a,
            response_b=resp_b,
        )
        prompt = []
        if SYSTEM_MSG:
            prompt.append({"role": "system", "content": SYSTEM_MSG})
        prompt.append({"role": "user", "content": user_msg})

        # Rubric already joined onto each row upstream (empty string on val).
        rubric_text = example.get("rubric", "") or ""

        return {
            "data_source": "helpsteer3_dedup",
            "prompt": prompt,
            "ability": "rm",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "domain": example.get("domain", ""),
                "individual_preference": example.get("individual_preference", []),
                "overall_preference": example["overall_preference"],
                "position_flipped": position_flipped,
                "feedback_mode": FEEDBACK_MODE,
                "rubric": rubric_text,
                "context_raw": context,
                "resp_a_raw": resp_a,
                "resp_b_raw": resp_b,
                "prompt_template_key": PROMPT_TEMPLATE_KEY,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="datasets/helpsteer3_dedup",
                        help="Output dir for parquet files.")
    parser.add_argument("--hf_repo", default="opsd-genrm/dedup_filtered_HS3",
                        help="Source HF Hub dataset.")
    parser.add_argument("--include_multiturn", action="store_true",
                        help="Include multiturn training samples")
    parser.add_argument("--include_format", action="store_true",
                        help="Wrap context messages and the two responses in chat-style tags.")
    parser.add_argument("--prompt_template", default="pair_rm",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        help="Which prompt template variant to use (default: pair_rm)")
    parser.add_argument("--feedback_mode", default=None,
                        choices=["reasoning", "feedback", "rubric", "none"],
                        help="Annotator feedback mode. 'rubric' reads the pre-attached "
                             "rubric column (train only has non-empty rubrics).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for position randomization")
    args = parser.parse_args()

    random.seed(args.seed)

    _tmpl = get_prompt_template(args.prompt_template)
    SYSTEM_MSG = _tmpl.student_sys_msg
    USER_MSG_TEMPLATE = _tmpl.student_user_msg
    FEEDBACK_MODE = None if args.feedback_mode == "none" else args.feedback_mode
    PROMPT_TEMPLATE_KEY = args.prompt_template

    print(f"Loading {args.hf_repo} from HF Hub...")
    dataset = datasets.load_dataset(args.hf_repo)

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    if not args.include_multiturn:
        train_dataset = train_dataset.filter(lambda x: len(x["context"]) == 1)
        test_dataset = test_dataset.filter(lambda x: len(x["context"]) == 1)

    train_dataset = train_dataset.map(
        function=make_map_fn("train", include_format=args.include_format),
        with_indices=True,
    )

    def add_uid(example, idx):
        example["uid"] = f"hs3_test_{idx}"
        return example

    test_dataset = test_dataset.map(
        function=make_map_fn("test", include_format=args.include_format),
        with_indices=True,
    )
    test_dataset = test_dataset.map(function=add_uid, with_indices=True)

    # Print one sample for verification
    sample = train_dataset[0]["prompt"]
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

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"\nTrain: {len(train_dataset)} examples -> {os.path.join(local_dir, 'train.parquet')}")
    print(f"Test:  {len(test_dataset)} examples -> {os.path.join(local_dir, 'test.parquet')}")
