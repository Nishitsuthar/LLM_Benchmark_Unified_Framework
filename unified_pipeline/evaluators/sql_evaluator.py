"""SQL detour evaluator.

Executes the model's SQL query against the imdb_controlled SQLite database,
then scores the result table using the same row-level F1 logic as TableF1Evaluator.

Flow:
  model response (SQL string)
      → execute against SQLite
      → result table as CSV string
      → TableF1Evaluator.evaluate(result_csv, ground_truth_path, expected_columns)

eval_status values:
  "evaluated"        — SQL ran and result was scored
  "sql_error: ..."   — SQL failed to execute (syntax error, bad column, etc.)
  "empty_result"     — SQL ran but returned zero rows
"""

import csv
import io
import math
import re
import sqlite3
from pathlib import Path

from unified_pipeline.base import BaseEvaluator, EvalResult
from unified_pipeline.evaluators.table_f1_evaluator import TableF1Evaluator


class SqlTableF1Evaluator(BaseEvaluator):
    def __init__(self) -> None:
        self._f1 = TableF1Evaluator()

    def evaluate(
        self,
        response_text: str,
        ground_truth_path: str,
        expected_columns: list[str],
        db_path: str = "",
    ) -> EvalResult:
        sql = self._extract_sql(response_text)

        try:
            result_csv = self._execute(sql, db_path)
        except Exception as exc:
            return EvalResult(0, 0.0, 0.0, 0.0, f"sql_error: {exc}")

        if not result_csv.strip():
            return EvalResult(0, 0.0, 0.0, 0.0, "empty_result")

        return self._f1.evaluate(result_csv, ground_truth_path, expected_columns)

    # ------------------------------------------------------------------

    def _extract_sql(self, text: str) -> str:
        # Strip markdown fences if model wrapped the query
        text = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).strip()
        text = text.replace("```", "").strip()
        return text

    def _execute(self, sql: str, db_path: str) -> str:
        con = sqlite3.connect(db_path)
        con.create_function("SQRT", 1, math.sqrt)
        try:
            cursor = con.execute(sql)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
        finally:
            con.close()

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(cols)
        writer.writerows(rows)
        return buf.getvalue()
