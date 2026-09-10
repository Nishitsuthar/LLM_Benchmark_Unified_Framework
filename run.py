"""Unified pipeline entry point.

Usage:
    python run.py --dataset imdb_controlled --model mistral_small
    python run.py --dataset music           --model gemini_flash
    python run.py --dataset uda_nqtext      --model deepseek_r1

An interactive menu always appears to choose the prompt. Only prompts
relevant to the selected dataset are shown (dataset-specific first,
then generic fallbacks).

--model accepts either a short alias from config.MODEL_ALIASES or a raw
provider model ID (e.g. meta-llama/llama-3.3-70b-instruct).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from unified_pipeline.config import EVIDENCE_BUILDERS, EVALUATORS
from unified_pipeline.model_router import ModelRouter
from unified_pipeline.reporter import Reporter


PROMPTS_DIR = Path("prompts")
RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the unified benchmark pipeline."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name matching a file in datasets/",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model alias or raw provider model ID",
    )

    parser.add_argument(
        "--task_ids",
        nargs="*",
        default=None,
        help="Optional subset of task IDs to run, e.g. --task_ids T01 T05",
    )

    return parser.parse_args()


def load_questions(questions_file: str) -> list[dict]:
    path = Path(questions_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {path}"
        )

    with path.open(
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    # Accept either a plain list or {"questions": [...]}
    return (
        data
        if isinstance(data, list)
        else data["questions"]
    )


def discover_prompts(
    dataset: str,
) -> list[tuple[str, Path]]:
    """Return (label, path) pairs — dataset-specific first, then generic."""

    found = []

    for p in sorted(PROMPTS_DIR.glob(f"{dataset}_*.txt")):
        found.append((f"{p.stem}  [dataset-specific]", p))

    for p in sorted(PROMPTS_DIR.glob("*.txt")):
        if not p.stem.startswith(dataset):
            found.append((f"{p.stem}  [generic]", p))

    return found


def select_prompt_interactive(
    options: list[tuple[str, Path]],
) -> tuple[str, str]:
    """Show a numbered menu and return (chosen_label, template_text)."""

    print("\n  Available prompts:")

    for i, (label, _) in enumerate(options, 1):
        print(f"    {i}. {label}")

    print(f"    0. Exit")

    while True:
        raw = input("  Select prompt [1]: ").strip() or "1"

        if raw == "0":
            print("  Exiting.")
            sys.exit(0)

        if raw.isdigit() and 1 <= int(raw) <= len(options):
            label, path = options[int(raw) - 1]
            return label, path.read_text(encoding="utf-8")

        print(f"  Please enter 0 to exit or a number between 1 and {len(options)}.")


def build_prompt(
    template: str,
    question: dict,
    evidence_text: str,
) -> str:
    return template.format(
        question=question.get(
            "text",
            "",
        ),
        evidence=evidence_text,
        answer_format=question.get(
            "answer_format",
            "CSV table",
        ),
    )


def _progress(
    current: int,
    total: int,
    q_id: str,
    stage: str,
) -> None:
    filled = int(
        30 * current / total
    )

    bar = (
        "█" * filled
        + "░" * (30 - filled)
    )

    pct = int(
        100 * current / total
    )

    sys.stdout.write(
        f"\r  [{bar}] "
        f"{pct:3d}%  "
        f"{current}/{total}  "
        f"{q_id} — "
        f"{stage:<25}"
    )

    sys.stdout.flush()


def main() -> None:
    args = parse_args()

    # --------------------------------------------------------------
    # Load dataset config
    # --------------------------------------------------------------

    dataset_path = Path(
        f"datasets/{args.dataset}.yaml"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {dataset_path}"
        )

    config = yaml.safe_load(
        dataset_path.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------------
    # Validate registry keys
    # --------------------------------------------------------------

    evidence_type = config[
        "evidence_type"
    ]

    eval_strategy = config[
        "eval_strategy"
    ]

    if (
        evidence_type
        not in EVIDENCE_BUILDERS
    ):
        raise KeyError(
            f"No evidence builder registered for "
            f"'{evidence_type}'. Check config.py."
        )

    if (
        eval_strategy
        not in EVALUATORS
    ):
        raise KeyError(
            f"No evaluator registered for "
            f"'{eval_strategy}'. Check config.py."
        )

    # --------------------------------------------------------------
    # Instantiate components
    # --------------------------------------------------------------

    builder = EVIDENCE_BUILDERS[
        evidence_type
    ]()

    evaluator = EVALUATORS[
        eval_strategy
    ]()

    router = ModelRouter(
        args.model
    )

    # --------------------------------------------------------------
    # Resolve prompt interactively
    # --------------------------------------------------------------

    options = discover_prompts(args.dataset)

    if not options:
        raise FileNotFoundError(
            f"No prompt files found in {PROMPTS_DIR}/ for dataset={args.dataset}. "
            "Add a file named {dataset}_{mode}_{style}.txt or {mode}_{style}.txt."
        )

    prompt_label, template = select_prompt_interactive(options)

    prompt_style_for_output = prompt_label.split("  ")[0]

    # Derive mode from the chosen prompt filename
    mode = "sql_detour" if "sql_detour" in prompt_style_for_output else "llm_only"

    reporter = Reporter(
        f"results/{args.model}/{args.dataset}_{mode}_{prompt_style_for_output}_metrics.csv"
    )

    # --------------------------------------------------------------
    # Load questions
    # --------------------------------------------------------------

    questions = load_questions(
        config["questions_file"]
    )

    if args.task_ids:
        selected = set(
            args.task_ids
        )

        questions = [
            q
            for q in questions
            if q.get("id")
            in selected
        ]

    print(
        f"\n  Dataset  : "
        f"{args.dataset}"
    )

    print(
        f"  Model    : "
        f"{args.model}"
    )

    print(
        f"  Mode     : "
        f"{mode}  |  "
        f"Prompt: {prompt_label}"
    )

    print(
        f"  Questions: "
        f"{len(questions)}"
    )

    print()

    run_start = time.time()

    f1_scores = []
    failed = []

    # --------------------------------------------------------------
    # Run benchmark questions
    # --------------------------------------------------------------

    for i, question in enumerate(
        questions,
        1,
    ):
        q_id = question.get(
            "id",
            "unknown",
        )

        try:
            # ------------------------------------------------------
            # Step 1 — build evidence
            # ------------------------------------------------------

            _progress(
                i,
                len(questions),
                q_id,
                "building evidence...",
            )

            question_config = {
                **config,
                "_source_pdf": question.get(
                    "source_pdf"
                ),
            }

            evidence = builder.build(
                question["text"],
                question_config,
            )

            # ------------------------------------------------------
            # Step 2 — build prompt and call model
            # ------------------------------------------------------

            _progress(
                i,
                len(questions),
                q_id,
                "calling model...    ",
            )

            prompt_text = build_prompt(
                template,
                question,
                evidence.text,
            )

            # Visual evidence builders can provide base64 image
            # data URLs through EvidenceResult.metadata["images"].
            # Text-only builders simply return no images.
            images = evidence.metadata.get(
                "images"
            )

            response = router.call(
                prompt_text,
                images=images,
            )

            # ------------------------------------------------------
            # Step 3 — evaluate
            # ------------------------------------------------------

            _progress(
                i,
                len(questions),
                q_id,
                "evaluating...       ",
            )

            result = evaluator.evaluate(
                response.final_text,
                str(
                    Path(
                        config[
                            "ground_truth_dir"
                        ]
                    )
                    / question[
                        "ground_truth_file"
                    ]
                ),
                question.get(
                    "expected_columns",
                    [],
                ),
            )

            # ------------------------------------------------------
            # Step 4 — write row immediately (crash safe)
            # ------------------------------------------------------

            ground_truth = (
                Path(
                    config[
                        "ground_truth_dir"
                    ]
                )
                / question[
                    "ground_truth_file"
                ]
            )

            reporter.write_row(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "mode": mode,
                    "prompt_style": (
                        prompt_style_for_output
                    ),
                    "question_id": q_id,
                    "question": question.get(
                        "text",
                        "",
                    ),
                    "ground_truth": (
                        ground_truth.read_text(
                            encoding="utf-8"
                        ).strip()
                    ),
                    "model_response": (
                        response.final_text
                    ),
                    "exact_match": (
                        result.content_exact_match
                    ),
                    "row_f1": (
                        result.row_f1
                    ),
                    "precision": (
                        result.precision
                    ),
                    "recall": (
                        result.recall
                    ),
                    "eval_status": (
                        result.evaluation_status
                    ),
                    "latency_seconds": (
                        response.latency_seconds
                    ),
                    "input_tokens": (
                        response.input_tokens
                    ),
                    "output_tokens": (
                        response.output_tokens
                    ),
                }
            )

            f1 = (
                result.row_f1
                if result.row_f1
                is not None
                else 0.0
            )

            f1_scores.append(
                f1
            )

            f1_str = (
                f"{f1:.3f}"
            )

            lat_str = (
                f"{response.latency_seconds:.1f}s"
            )

            exact = (
                "✓"
                if result.content_exact_match
                else "✗"
            )

            _progress(
                i,
                len(questions),
                q_id,
                (
                    f"done  "
                    f"f1={f1_str} "
                    f"{exact} "
                    f"{lat_str}"
                ),
            )

            print()

        except Exception as exc:
            failed.append(
                q_id
            )

            reporter.write_row(
                {
                    "dataset": args.dataset,
                    "model": args.model,
                    "mode": mode,
                    "prompt_style": (
                        prompt_style_for_output
                    ),
                    "question_id": q_id,
                    "question": question.get(
                        "text",
                        "",
                    ),
                    "ground_truth": None,
                    "model_response": None,
                    "exact_match": 0,
                    "row_f1": None,
                    "precision": None,
                    "recall": None,
                    "eval_status": (
                        f"error: "
                        f"{type(exc).__name__}"
                    ),
                    "latency_seconds": None,
                    "input_tokens": None,
                    "output_tokens": None,
                }
            )

            _progress(
                i,
                len(questions),
                q_id,
                (
                    f"FAILED: "
                    f"{type(exc).__name__}"
                ),
            )

            print(
                f"\n  ! {q_id} "
                f"error: {exc}"
            )

    # --------------------------------------------------------------
    # Run summary
    # --------------------------------------------------------------

    elapsed = (
        time.time()
        - run_start
    )

    avg_f1 = (
        sum(f1_scores)
        / len(f1_scores)
        if f1_scores
        else 0.0
    )

    exact_n = sum(
        1
        for f in f1_scores
        if f == 1.0
    )

    print()

    print(
        "  ─" * 25
    )

    print(
        f"  Finished "
        f"{len(questions)} questions "
        f"in {elapsed:.1f}s"
    )

    print(
        f"  Avg F1   : "
        f"{avg_f1:.3f}"
    )

    print(
        f"  Exact    : "
        f"{exact_n}/"
        f"{len(questions)}"
    )

    if failed:
        print(
            f"  Failed   : "
            f"{len(failed)} — "
            f"{', '.join(failed)}"
        )

    print(
        f"  Results  : "
        f"results/{args.model}/{args.dataset}_{mode}_{prompt_style_for_output}_metrics.csv"
    )

    print()


if __name__ == "__main__":
    main()