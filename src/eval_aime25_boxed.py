#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估 AIME25 结果的脚本。

功能：
1. 读取结果目录或 zip 压缩包（每道题对应一个数字文件夹，如 0/0.json）。
2. 仅当最后一步中 "answer" == True 时，才从最后一步的 base_model_step 中提取答案。
3. 从最后一个 \boxed{...} 中做稳健提取，支持嵌套花括号。
4. 读取 aime25.parquet 中的标准答案并计算正确率。
5. 输出逐题明细 CSV 和汇总 TXT。

用法示例：
python eval_aime25_boxed.py \
    --results /path/to/aime25.zip \
    --dataset /path/to/aime25.parquet \
    --out_dir /path/to/eval_out
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DIGIT_JSON_RE = re.compile(r"^(\d+)/(?:0\.json)$")


@dataclass
class EvalRow:
    problem_id: int
    json_path: str
    answer_flag: Optional[bool]
    gold_answer: str
    extracted_boxed_raw: Optional[str]
    normalized_pred: Optional[str]
    is_correct: bool
    status: str
    problem: str


class ResultReader:
    def list_json_paths(self) -> list[str]:
        raise NotImplementedError

    def read_json(self, path: str):
        raise NotImplementedError


class ZipResultReader(ResultReader):
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        self.zf = zipfile.ZipFile(zip_path, "r")

    def list_json_paths(self) -> list[str]:
        paths = []
        for name in self.zf.namelist():
            if DIGIT_JSON_RE.fullmatch(name):
                paths.append(name)
        return sorted(paths, key=lambda p: int(p.split("/")[0]))

    def read_json(self, path: str):
        return json.loads(self.zf.read(path))


class DirResultReader(ResultReader):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def list_json_paths(self) -> list[str]:
        paths = []
        for subdir in self.root_dir.iterdir():
            if not subdir.is_dir() or not subdir.name.isdigit():
                continue
            json_path = subdir / "0.json"
            if json_path.exists():
                paths.append(str(json_path.relative_to(self.root_dir)))
        return sorted(paths, key=lambda p: int(Path(p).parts[0]))

    def read_json(self, path: str):
        return json.loads((self.root_dir / path).read_text(encoding="utf-8"))


def build_reader(results_path: Path) -> ResultReader:
    if results_path.is_file() and results_path.suffix.lower() == ".zip":
        return ZipResultReader(results_path)
    if results_path.is_dir():
        return DirResultReader(results_path)
    raise FileNotFoundError(f"未找到可用的结果路径: {results_path}")


def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    """从文本中提取最后一个 \boxed{...} 的内容，支持嵌套花括号。"""
    if not isinstance(text, str) or not text:
        return None

    positions = []
    start = 0
    while True:
        idx = text.find(r"\boxed", start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + len(r"\boxed")

    if not positions:
        return None

    idx = positions[-1] + len(r"\boxed")
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text):
        return None

    # 标准形式：\boxed{...}
    if text[idx] == "{":
        depth = 0
        content_start = idx + 1
        for j in range(idx, len(text)):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[content_start:j]
        return None

    # 兜底：若有人写成 \boxed 123，则取后面连续的非空白 token
    j = idx
    while j < len(text) and not text[j].isspace():
        j += 1
    token = text[idx:j].strip()
    return token or None


def strip_outer_wrappers(s: str) -> str:
    """重复剥离最外层成对的 (), {}, []。"""
    pairs = {"(": ")", "{": "}", "[": "]"}

    def is_single_wrapped(x: str, left: str, right: str) -> bool:
        if not (x.startswith(left) and x.endswith(right)):
            return False
        depth = 0
        for i, ch in enumerate(x):
            if ch == left:
                depth += 1
            elif ch == right:
                depth -= 1
                if depth == 0 and i != len(x) - 1:
                    return False
        return depth == 0

    changed = True
    while changed and s:
        changed = False
        for left, right in pairs.items():
            if is_single_wrapped(s, left, right):
                s = s[1:-1].strip()
                changed = True
    return s


def normalize_pred(expr: Optional[str]) -> Optional[str]:
    """
    将 boxed 内部内容规范化为适合和 AIME 标准答案比较的字符串。

    规则尽量保守：
    - 只在明确可归一为整数时，才转成整数串。
    - 对 \frac{a}{b} / a/b，仅当可以整除时转整数，否则保留分数形式。
    - 去掉一些明显不影响整数答案的 LaTeX 外壳，如 ^\\circ。
    """
    if expr is None:
        return None

    s = expr.strip()
    if not s:
        return None

    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    s = strip_outer_wrappers(s)

    # 去掉常见无关后缀，例如角度符号
    s = re.sub(r"\^\{?\\circ\}?$", "", s)
    s = re.sub(r"\^\{?circ\}?$", "", s)
    s = re.sub(r"\\text\{[^{}]*\}$", "", s)
    s = strip_outer_wrappers(s)

    if re.fullmatch(r"[+-]?\d+", s):
        return str(int(s))

    frac_patterns = [
        re.compile(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}$"),
        re.compile(r"([+-]?\d+)/([+-]?\d+)$"),
    ]
    for pat in frac_patterns:
        m = pat.fullmatch(s)
        if m:
            num = int(m.group(1))
            den = int(m.group(2))
            if den == 0:
                return s
            if num % den == 0:
                return str(num // den)
            return f"{num}/{den}"

    return s


def load_dataset(dataset_path: Path):
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "缺少 pandas。请先安装: pip install pandas pyarrow"
        ) from e

    try:
        df = pd.read_parquet(dataset_path)
    except Exception as e:
        raise RuntimeError(
            "读取 parquet 失败。请确认已安装 pyarrow: pip install pyarrow"
        ) from e

    required_cols = {"id", "problem", "answer"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"parquet 缺少必要字段: {sorted(missing)}")

    gold_map = {}
    for _, row in df.iterrows():
        pid = int(row["id"])
        gold_map[pid] = {
            "problem": str(row["problem"]),
            "answer": str(row["answer"]).strip(),
        }
    return gold_map


def evaluate(results_path: Path, dataset_path: Path) -> list[EvalRow]:
    gold_map = load_dataset(dataset_path)
    reader = build_reader(results_path)
    json_paths = reader.list_json_paths()

    rows: list[EvalRow] = []
    for json_path in json_paths:
        problem_id = int(Path(json_path).parts[0])
        if problem_id not in gold_map:
            rows.append(
                EvalRow(
                    problem_id=problem_id,
                    json_path=json_path,
                    answer_flag=None,
                    gold_answer="<missing in dataset>",
                    extracted_boxed_raw=None,
                    normalized_pred=None,
                    is_correct=False,
                    status="dataset_missing",
                    problem="",
                )
            )
            continue

        data = reader.read_json(json_path)
        if not isinstance(data, list) or not data:
            rows.append(
                EvalRow(
                    problem_id=problem_id,
                    json_path=json_path,
                    answer_flag=None,
                    gold_answer=gold_map[problem_id]["answer"],
                    extracted_boxed_raw=None,
                    normalized_pred=None,
                    is_correct=False,
                    status="invalid_json_content",
                    problem=gold_map[problem_id]["problem"],
                )
            )
            continue

        last_step = data[-1]
        answer_flag = last_step.get("answer", None)
        gold_answer = gold_map[problem_id]["answer"]
        problem_text = gold_map[problem_id]["problem"]

        extracted_boxed_raw = None
        normalized_pred = None
        is_correct = False
        status = "not_answered"

        if answer_flag is True:
            extracted_boxed_raw = extract_last_boxed(last_step.get("base_model_step"))
            normalized_pred = normalize_pred(extracted_boxed_raw)
            if normalized_pred is None:
                status = "answer_true_but_no_boxed"
            else:
                is_correct = normalized_pred == gold_answer
                status = "correct" if is_correct else "wrong"
        elif answer_flag is False:
            status = "answer_false_skip_extract"
        else:
            status = "missing_answer_flag"

        rows.append(
            EvalRow(
                problem_id=problem_id,
                json_path=json_path,
                answer_flag=answer_flag,
                gold_answer=gold_answer,
                extracted_boxed_raw=extracted_boxed_raw,
                normalized_pred=normalized_pred,
                is_correct=is_correct,
                status=status,
                problem=problem_text,
            )
        )

    # 补齐数据集中有但结果里不存在的题目
    existing_ids = {r.problem_id for r in rows}
    for problem_id in sorted(gold_map):
        if problem_id not in existing_ids:
            rows.append(
                EvalRow(
                    problem_id=problem_id,
                    json_path="<missing_result>",
                    answer_flag=None,
                    gold_answer=gold_map[problem_id]["answer"],
                    extracted_boxed_raw=None,
                    normalized_pred=None,
                    is_correct=False,
                    status="missing_result",
                    problem=gold_map[problem_id]["problem"],
                )
            )

    rows.sort(key=lambda x: x.problem_id)
    return rows


def write_outputs(rows: list[EvalRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "aime25_eval_details.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "problem_id",
                "json_path",
                "answer_flag",
                "gold_answer",
                "extracted_boxed_raw",
                "normalized_pred",
                "is_correct",
                "status",
                "problem",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.problem_id,
                    r.json_path,
                    r.answer_flag,
                    r.gold_answer,
                    r.extracted_boxed_raw,
                    r.normalized_pred,
                    r.is_correct,
                    r.status,
                    r.problem,
                ]
            )

    total = len(rows)
    answered_true = sum(r.answer_flag is True for r in rows)
    answered_false = sum(r.answer_flag is False for r in rows)
    correct = sum(r.is_correct for r in rows)
    extracted = sum(r.normalized_pred is not None for r in rows)
    missing_result = sum(r.status == "missing_result" for r in rows)
    accuracy_all = correct / total if total else 0.0
    accuracy_answered = correct / answered_true if answered_true else 0.0

    txt_path = out_dir / "aime25_eval_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"数据集总题数                 : {total}\n")
        f.write(f"存在结果文件的题数           : {total - missing_result}\n")
        f.write(f"answer == True 的题数        : {answered_true}\n")
        f.write(f"answer == False 的题数       : {answered_false}\n")
        f.write(f"成功提取 boxed 答案的题数    : {extracted}\n")
        f.write(f"正确题数                     : {correct}\n")
        f.write(f"总体准确率（correct/total）  : {accuracy_all:.4%}\n")
        f.write(f"作答子集准确率（correct/True）: {accuracy_answered:.4%}\n")
        f.write("=" * 80 + "\n")

    print("=" * 80)
    print(f"数据集总题数                 : {total}")
    print(f"存在结果文件的题数           : {total - missing_result}")
    print(f"answer == True 的题数        : {answered_true}")
    print(f"answer == False 的题数       : {answered_false}")
    print(f"成功提取 boxed 答案的题数    : {extracted}")
    print(f"正确题数                     : {correct}")
    print(f"总体准确率（correct/total）  : {accuracy_all:.4%}")
    print(f"作答子集准确率（correct/True）: {accuracy_answered:.4%}")
    print("=" * 80)
    print(f"\n逐题明细已保存到: {csv_path}")
    print(f"汇总结果已保存到: {txt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 AIME25 结果正确率")
    parser.add_argument(
        "--results",
        type=Path,
        # required=True,
        default="router_result/aime25",
        help="结果目录或 zip 压缩包路径",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        # required=True,
        default="data/aime25.parquet",
        help="aime25.parquet 路径",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("router_result/aime25/aime25_eval_out"),
        help="评估输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = evaluate(args.results, args.dataset)
    write_outputs(rows, args.out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
