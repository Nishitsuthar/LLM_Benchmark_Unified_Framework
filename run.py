"""Unified pipeline entry point.

Usage:
    python run.py --dataset imdb_controlled --model mistral_small
    python run.py --dataset music           --model gemini_flash
    python run.py --dataset uda_nqtext      --model deepseek_r1

IMDb-20 uses its fixed zero-shot prompt. Other datasets show an interactive menu. Only prompts
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
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

import yaml

from unified_pipeline.config import EVIDENCE_BUILDERS, EVALUATORS
from unified_pipeline.model_router import ModelRouter
from unified_pipeline.reporter import Reporter


PROMPTS_DIR = Path("prompts")
RESULTS_DIR = Path("results")
IMDB_DATASETS = {"imdb_20", "imdb_20_final45"}
IMDB_MODELS = {
    "gemini_flash": "google/gemini-3.1-flash-lite",
    "mistral_small": "mistralai/mistral-small-2603",
}


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

    parser.add_argument(
        "--confirm-full-run", action="store_true",
        help="IMDb-20: explicitly allow more than one question after smoke-test review",
    )
    args = parser.parse_args()
    if args.dataset in IMDB_DATASETS:
        aliases = {value: key for key, value in IMDB_MODELS.items()}
        args.model = aliases.get(args.model, args.model)
        if args.model not in IMDB_MODELS:
            parser.error("imdb_20 supports gemini_flash or mistral_small for this workflow")
        if not args.confirm_full_run and (not args.task_ids or len(args.task_ids) != 1):
            parser.error("First use --task_ids F1; after review, use --confirm-full-run")
    return args


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

    all_datasets = {p.stem for p in Path("datasets").glob("*.yaml")}

    found = []

    for p in sorted(PROMPTS_DIR.glob(f"{dataset}_*.txt")):
        found.append((f"{p.stem}  [dataset-specific]", p))

    for p in sorted(PROMPTS_DIR.glob("*.txt")):
        if not any(p.stem.startswith(ds) for ds in all_datasets):
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
    sql_dialect: str = "sqlite",
    schema: str = "",
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
        sql_dialect=sql_dialect,
        schema=schema,
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
    imdb_run = args.dataset in IMDB_DATASETS

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

    router = ModelRouter(args.model)
    if imdb_run:
        # Count retries in our router rather than hiding additional SDK requests.
        router.client = router.client.with_options(max_retries=0)

    # --------------------------------------------------------------
    # Resolve prompt interactively
    # --------------------------------------------------------------

    options = discover_prompts(args.dataset)

    if not options:
        raise FileNotFoundError(
            f"No prompt files found in {PROMPTS_DIR}/ for dataset={args.dataset}. "
            "Add a file named {dataset}_{mode}_{style}.txt or {mode}_{style}.txt."
        )

    if imdb_run:
        prompt_path = PROMPTS_DIR / f"{args.dataset}_llm_only_zero_shot.txt"
        prompt_label = prompt_path.stem
        template = prompt_path.read_text(encoding="utf-8")
    else:
        prompt_label, template = select_prompt_interactive(options)

    prompt_style_for_output = prompt_label.split("  ")[0]

    # Derive mode from the chosen prompt filename
    mode = "sql_detour" if "sql_detour" in prompt_style_for_output else "llm_only"

    # Build output stem — dataset-specific prompts already contain the dataset
    # name; generic prompts need it prepended so files don't collide across datasets
    output_stem = (
        prompt_style_for_output
        if prompt_style_for_output.startswith(args.dataset)
        else f"{args.dataset}_{prompt_style_for_output}"
    )

    # Prepare SQL data only for SQL-detour mode on CSV-backed datasets
    subset_df = None
    db_path_for_sql = None
    schema = ""

    if mode == "sql_detour":
        if config.get("data_file"):
            subset_df = pd.read_csv(config["data_file"])
            schema = router.build_sql_schema(subset_df)
        elif config.get("sql_db_file"):
            db_path = Path(config["sql_db_file"])
            if not db_path.exists():
                raise FileNotFoundError(
                    f"SQL database not found: {db_path}\n"
                    f"Please create it and place it at that path before running sql_detour."
                )
            db_path_for_sql = str(db_path)
            schema = router.build_sql_schema_from_db(db_path_for_sql)

    output_dir = RESULTS_DIR / args.model
    if imdb_run:
        # Separate smoke and full runs; never append into a previous run's metrics.
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = output_dir / args.dataset / run_id
    reporter = Reporter(str(output_dir / f"{output_stem}_metrics.csv"))
    raw_path = output_dir / f"{output_stem}_raw.jsonl"

    def raw_message_text() -> str | None:
        raw = router.last_raw_response or {}
        choices = raw.get("choices") or []
        if not choices:
            return None
        content = (choices[0].get("message") or {}).get("content")
        return content if isinstance(content, str) or content is None else json.dumps(content, ensure_ascii=False)

    def save_raw(question_id: str) -> None:
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "question_id": question_id, "model_id": router.model_name,
                "response": router.last_raw_response,
            }, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------
    # Load questions
    # --------------------------------------------------------------

    questions = load_questions(
        config["questions_file"]
    )

    if imdb_run and args.task_ids:
        unknown = set(args.task_ids) - {q.get("id") for q in questions}
        if unknown:
            raise ValueError(f"Unknown IMDb question IDs: {sorted(unknown)}")

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
    binary_scores = []
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

        response = None
        images = None
        successes_before = router.api_successes if imdb_run else 0
        failures_before = router.api_failures if imdb_run else 0
        raw_saved = False
        ground_truth_text = None
        question_start = time.perf_counter()
        if imdb_run:
            router.last_raw_response = None
        try:
            if imdb_run:
                ground_truth_text = (Path(config["ground_truth_dir"]) / question["ground_truth_file"]).read_text(encoding="utf-8").strip()
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
                "question_doc_file": question.get("doc_file", ""),
                "_question_id": question.get("id"),
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
                sql_dialect=config.get("sql_dialect", "sqlite"),
                schema=schema,
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

            if imdb_run:
                save_raw(q_id)
                raw_saved = True
            evaluation_text = response.final_text

            if mode == "sql_detour":
                generated_sql = router.extract_sql(
                    response.final_text
                    )

                if db_path_for_sql:
                    sql_result_csv, sql_error, actual_columns, row_count = (
                        router.execute_sql_from_db(
                            generated_sql,
                            db_path_for_sql,
                        )
                    )
                else:
                    sql_result_csv, sql_error, actual_columns, row_count = (
                        router.execute_sql(
                            generated_sql,
                            subset_df,
                        )
                    )

                if sql_error:
                    raise RuntimeError(
                        f"SQL execution failed: {sql_error}"
                    )

                evaluation_text = sql_result_csv

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
                evaluation_text,
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

            if imdb_run:
                # Preserve the evaluator's conservative normalization; use binary scoring.
                result.evaluation_status = "evaluated_exact" if result.content_exact_match else "evaluated_incorrect"
                result.row_f1 = result.precision = result.recall = None

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
                    **({
                        "model_id": router.model_name,
                        "valid_question": True,
                        "category": question.get("category", ""),
                        "automatic_score": result.content_exact_match,
                        "evaluation_status": result.evaluation_status,
                        "scored_response": response.final_text,
                        "image_count": len(images or []),
                        "api_successes": router.api_successes - successes_before,
                        "api_failures": router.api_failures - failures_before,
                        "error": "",
                        "reasoning_text": response.reasoning_text,
                        "finish_reason": response.finish_reason,
                        "raw_response_file": str(raw_path),
                    } if imdb_run else {}),
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
                        raw_message_text() if imdb_run else evaluation_text
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

            if imdb_run:
                binary_scores.append(result.content_exact_match)

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
                    + (f"exact_match={result.content_exact_match} " if imdb_run else f"f1={f1_str} ")
                    +
                    f"{exact} "
                    f"{lat_str}"
                ),
            )

            print()

        except Exception as exc:
            if imdb_run and not raw_saved and router.last_raw_response is not None:
                save_raw(q_id)
            failed.append(
                q_id
            )

            reporter.write_row(
                {
                    **({
                        "model_id": router.model_name,
                        "valid_question": False,
                        "category": question.get("category", ""),
                        "automatic_score": 0,
                        "evaluation_status": f"error: {type(exc).__name__}",
                        "scored_response": response.final_text if response else None,
                        "image_count": len(images or []),
                        "api_successes": router.api_successes - successes_before,
                        "api_failures": router.api_failures - failures_before,
                        "error": f"{type(exc).__name__}: {exc}",
                        "reasoning_text": response.reasoning_text if response else None,
                        "finish_reason": response.finish_reason if response else None,
                        "raw_response_file": str(raw_path),
                    } if imdb_run else {}),
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
                    "ground_truth": ground_truth_text if imdb_run else None,
                    "model_response": raw_message_text() if imdb_run else None,
                    "exact_match": 0,
                    "row_f1": None,
                    "precision": None,
                    "recall": None,
                    "eval_status": (
                        f"error: "
                        f"{type(exc).__name__}"
                    ),
                    "latency_seconds": (response.latency_seconds if response else round(time.perf_counter() - question_start, 3)) if imdb_run else None,
                    "input_tokens": response.input_tokens if imdb_run and response else None,
                    "output_tokens": response.output_tokens if imdb_run and response else None,
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

    if imdb_run:
        correct = sum(binary_scores)
        valid = len(binary_scores)
        accuracy = correct / valid if valid else None
        summary = {
            "dataset": args.dataset, "model_id": router.model_name,
            "correct": correct, "valid_questions": valid,
            "attempted_questions": len(questions), "errors": len(failed),
            "accuracy": accuracy,
            "successful_api_calls": router.api_successes,
            "failed_api_calls": router.api_failures,
            "denominator_policy": "Successfully evaluated responses; execution errors excluded; empty or non-exact answers score 0",
        }
        (output_dir / f"{output_stem}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        )
        print(f"  ACCURACY : {accuracy:.3f} ({correct}/{valid} valid questions)" if valid else "  ACCURACY : N/A (no valid questions)")
    else:
        print(f"  Avg F1   : {avg_f1:.3f}")
        print(f"  Exact    : {exact_n}/{len(questions)}")

    if failed:
        print(
            f"  Failed   : "
            f"{len(failed)} — "
            f"{', '.join(failed)}"
        )

    print(
        f"  Results  : "
        f"{reporter.path}"
    )

    print()


if __name__ == "__main__":
    main()
