"""OpenAI-compatible model client.

Ported from run_experiment.py. Handles:
- Provider-neutral API calls via MODEL_* env vars
- Retry logic with exponential back-off
- <think> tag separation for reasoning models
- Degeneration detection (repetitive output)
- Multimodal image input for visual evidence
- SQL-detour: extract + validate + execute generated SQL
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from unified_pipeline.base import ModelResponse
from unified_pipeline.config import MODEL_ALIASES


SQL_TABLE_NAME = "imdb_structured_subset"


class ModelCallError(RuntimeError):
    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class ModelRouter:
    def __init__(
        self,
        model_alias: str,
        request_timeout: int = 600,
        max_retries: int = 3,
        retry_sleep_seconds: int = 10,
        max_output_tokens: int = 16384,
        token_parameter: str = "max_tokens",
        temperature: float = 0.0,
        top_p: float | None = None,
        seed: int | None = None,
        extra_body: dict | None = None,
    ):
        load_dotenv()

        self.model_name = MODEL_ALIASES.get(
            model_alias,
            model_alias,
        )

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.max_output_tokens = max_output_tokens
        self.token_parameter = token_parameter
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.extra_body = extra_body or {}

        api_key = os.getenv("MODEL_API_KEY")
        base_url = os.getenv("MODEL_BASE_URL", "")

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
        }

        if base_url:
            client_kwargs["base_url"] = base_url.rstrip("/")

        self.client = OpenAI(**client_kwargs)

    def call(
        self,
        prompt_text: str,
        images: list[str] | None = None,
    ) -> ModelResponse:
        """Call the model and return a normalised response.

        Parameters
        ----------
        prompt_text:
            Text prompt sent to the model.

        images:
            Optional list of base64 image data URLs. When supplied,
            the request is sent as multimodal text + image content.

        Existing text-only datasets remain backward-compatible because
        callers can continue using:

            router.call(prompt_text)

        Visual datasets can use:

            router.call(prompt_text, images=images)
        """

        last_error: Exception | None = None
        start = time.perf_counter()

        # --------------------------------------------------------------
        # Prepare user message content
        # --------------------------------------------------------------

        if images:
            user_content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": prompt_text,
                }
            ]

            for image_url in images:
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    }
                )

            message_content: Any = user_content

        else:
            # Preserve original behaviour for CSV, RAG, prose, etc.
            message_content = prompt_text

        # --------------------------------------------------------------
        # Model call with retry logic
        # --------------------------------------------------------------

        for attempt in range(
            1,
            max(1, self.max_retries) + 1,
        ):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": message_content,
                        }
                    ],
                    "timeout": self.request_timeout,
                    "stream": False,
                }

                if self.max_output_tokens > 0:
                    kwargs[self.token_parameter] = (
                        self.max_output_tokens
                    )

                if self.temperature is not None:
                    kwargs["temperature"] = self.temperature

                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p

                if self.seed is not None:
                    kwargs["seed"] = self.seed

                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body

                response = self.client.chat.completions.create(
                    **kwargs
                )

                latency = round(
                    time.perf_counter() - start,
                    3,
                )

                return self._normalize(
                    response,
                    latency,
                )

            except Exception as exc:
                last_error = exc

                if (
                    attempt >= self.max_retries
                    or not self._is_retryable(str(exc))
                ):
                    break

                time.sleep(
                    self.retry_sleep_seconds * attempt
                )

        raise ModelCallError(
            (
                str(last_error)
                if last_error
                else "Unknown model-call failure"
            ),
            attempts=attempt,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(
        self,
        response: Any,
        latency: float,
    ) -> ModelResponse:
        choices = getattr(
            response,
            "choices",
            [],
        ) or []

        choice = choices[0] if choices else {}

        message = getattr(
            choice,
            "message",
            {},
        ) or {}

        content = self._to_text(
            getattr(
                message,
                "content",
                "",
            )
        )

        separate_reasoning = self._to_text(
            getattr(
                message,
                "reasoning_content",
                None,
            )
            or getattr(
                message,
                "reasoning",
                None,
            )
            or ""
        )

        final_text, tagged_reasoning = (
            self._split_think_tags(content)
        )

        reasoning_parts = [
            part
            for part in [
                separate_reasoning,
                tagged_reasoning,
            ]
            if part.strip()
        ]

        reasoning_text = "\n\n".join(
            dict.fromkeys(reasoning_parts)
        )

        usage = getattr(
            response,
            "usage",
            {},
        ) or {}

        input_tokens = (
            getattr(
                usage,
                "prompt_tokens",
                None,
            )
            or getattr(
                usage,
                "input_tokens",
                None,
            )
        )

        output_tokens = (
            getattr(
                usage,
                "completion_tokens",
                None,
            )
            or getattr(
                usage,
                "output_tokens",
                None,
            )
        )

        finish_reason = str(
            getattr(
                choice,
                "finish_reason",
                "",
            )
            or ""
        )

        return ModelResponse(
            final_text=final_text.strip(),
            reasoning_text=reasoning_text.strip(),
            finish_reason=finish_reason,
            input_tokens=(
                int(input_tokens)
                if input_tokens is not None
                else None
            ),
            output_tokens=(
                int(output_tokens)
                if output_tokens is not None
                else None
            ),
            latency_seconds=latency,
        )

    @staticmethod
    def _to_text(
        value: Any,
    ) -> str:
        if value is None:
            return ""

        if isinstance(
            value,
            str,
        ):
            return value

        if isinstance(
            value,
            list,
        ):
            return "".join(
                ModelRouter._to_text(item)
                for item in value
            )

        if isinstance(
            value,
            dict,
        ):
            for key in [
                "text",
                "content",
                "value",
            ]:
                if key in value:
                    return ModelRouter._to_text(
                        value[key]
                    )

        return str(value)

    @staticmethod
    def _split_think_tags(
        content: str,
    ) -> tuple[str, str]:
        if not content:
            return "", ""

        match = re.match(
            r"(?is)^\s*<think>(.*?)</think>\s*(.*)$",
            content,
        )

        if match:
            return (
                match.group(2).strip(),
                match.group(1).strip(),
            )

        match = re.match(
            r"(?is)^\s*<think>(.*)$",
            content,
        )

        if match:
            return (
                "",
                match.group(1).strip(),
            )

        return (
            content.strip(),
            "",
        )

    @staticmethod
    def _is_retryable(
        error_message: str,
    ) -> bool:
        non_retryable = [
            "400",
            "401",
            "403",
            "404",
            "405",
            "bad request",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "model not found",
            "context length",
        ]

        lower = error_message.lower()

        return not any(
            marker in lower
            for marker in non_retryable
        )

    # ------------------------------------------------------------------
    # Degeneration detection (used by run.py)
    # ------------------------------------------------------------------

    @staticmethod
    def is_degenerate(
        text: str,
        max_duplicate_ratio: float = 0.50,
        min_lines: int = 10,
        max_identical_lines: int = 20,
    ) -> bool:
        lines = [
            re.sub(
                r"\s+",
                " ",
                line.strip(),
            )
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return False

        counts = Counter(lines)

        duplicate_ratio = (
            len(lines) - len(counts)
        ) / len(lines)

        return (
            (
                len(lines) >= min_lines
                and duplicate_ratio
                >= max_duplicate_ratio
            )
            or max(counts.values())
            >= max_identical_lines
        )

    # ------------------------------------------------------------------
    # SQL-detour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_sql(
        text: str,
    ) -> str:
        if not text or not text.strip():
            return ""

        fenced = re.findall(
            r"```(?:sql|sqlite)?\s*(.*?)```",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        candidates = (
            list(reversed(fenced))
            + [text]
        )

        select_pattern = re.compile(
            r"(?im)^[ \t]*SELECT\b"
        )

        with_pattern = re.compile(
            r"(?im)^[ \t]*WITH"
            r"(?:[ \t]+RECURSIVE)?"
            r"[ \t\r\n]+"
            r"[A-Za-z_][A-Za-z0-9_]*"
            r"(?:[ \t]*\([^)]*\))?"
            r"[ \t\r\n]+AS"
            r"[ \t\r\n]*\("
        )

        for candidate in candidates:
            candidate = re.sub(
                (
                    r"^\s*"
                    r"(sql\s*query|query|sqlite\s*query|answer)"
                    r"\s*:\s*"
                ),
                "",
                candidate.strip(),
                flags=re.IGNORECASE,
            )

            matches = [
                match
                for match in [
                    select_pattern.search(candidate),
                    with_pattern.search(candidate),
                ]
                if match
            ]

            if not matches:
                continue

            start = min(
                matches,
                key=lambda match: match.start(),
            ).start()

            sql = candidate[
                start:
            ].strip()

            semi = sql.find(";")

            if semi >= 0:
                sql = sql[
                    : semi + 1
                ]

            return sql.strip()

        return ""

    @staticmethod
    def validate_sql(
        sql: str,
    ) -> tuple[bool, str]:
        cleaned = re.sub(
            r"/\*.*?\*/",
            "",
            sql,
            flags=re.DOTALL,
        )

        cleaned = re.sub(
            r"--.*?$",
            "",
            cleaned,
            flags=re.MULTILINE,
        ).strip()

        if not cleaned:
            return (
                False,
                "empty_generated_sql",
            )

        if not re.match(
            r"^(WITH|SELECT)\b",
            cleaned,
            re.IGNORECASE,
        ):
            return (
                False,
                "not_a_select_query",
            )

        blocked = [
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bDELETE\b",
            r"\bDROP\b",
            r"\bCREATE\b",
            r"\bALTER\b",
            r"\bATTACH\b",
            r"\bPRAGMA\b",
        ]

        for pattern in blocked:
            if re.search(
                pattern,
                cleaned,
                re.IGNORECASE,
            ):
                return (
                    False,
                    "non_read_only_sql_detected",
                )

        without_semi = (
            cleaned[:-1]
            if cleaned.endswith(";")
            else cleaned
        )

        if ";" in without_semi:
            return (
                False,
                "multiple_sql_statements_detected",
            )

        return (
            True,
            "valid",
        )
    
    @staticmethod
    def build_sql_schema(subset_df: pd.DataFrame) -> str:
        with sqlite3.connect(":memory:") as conn:
            subset_df.to_sql(
                SQL_TABLE_NAME,
                conn,
                index=False,
                if_exists="replace",
            )

            rows = conn.execute(
                f'PRAGMA table_info("{SQL_TABLE_NAME}")'
            ).fetchall()

        columns = "\n".join(
            f"{row[1]} {row[2]}"
            for row in rows
        )

        return f"Table: {SQL_TABLE_NAME}\nColumns:\n{columns}"

    @staticmethod
    def execute_sql(
        sql: str,
        subset_df: pd.DataFrame,
    ) -> tuple[str, str, list[str], int]:
        with sqlite3.connect(
            ":memory:"
        ) as conn:
            subset_df.to_sql(
                SQL_TABLE_NAME,
                conn,
                index=False,
                if_exists="replace",
            )

            conn.create_function(
                "SQRT",
                1,
                lambda value: (
                    math.sqrt(value)
                    if (
                        value is not None
                        and value >= 0
                    )
                    else None
                ),
            )

            conn.create_function(
                "CEIL",
                1,
                lambda value: (
                    math.ceil(value)
                    if value is not None
                    else None
                ),
            )

            conn.create_function(
                "FLOOR",
                1,
                lambda value: (
                    math.floor(value)
                    if value is not None
                    else None
                ),
            )

            conn.create_function(
                "POWER",
                2,
                lambda base, exponent: (
                    math.pow(
                        base,
                        exponent,
                    )
                    if (
                        base is not None
                        and exponent is not None
                    )
                    else None
                ),
            )

            try:
                result_df = pd.read_sql_query(
                    sql,
                    conn,
                )

            except Exception as exc:
                return (
                    "",
                    str(exc),
                    [],
                    0,
                )

        return (
            result_df.to_csv(
                index=False
            ),
            "",
            list(result_df.columns),
            len(result_df),
        )