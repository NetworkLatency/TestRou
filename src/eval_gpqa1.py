#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import zipfile
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError as e:
    raise SystemExit(
        "缺少 pandas。请先安装：pip install pandas pyarrow"
    ) from e


ANSWER_PATTERNS = [
    re.compile(r"\\boxed\s*\{\s*\{?\s*([ABCD])\s*\}?\s*\}", re.IGNORECASE),
    re.compile(r"\*\*\s*([ABCD])\s*\*\*"),
    re.compile(
        r"(?i)(?:final answer|correct answer|the answer is|therefore[,]?\s*the correct answer is|"
        r"thus[,]?\s*the correct answer is|therefore[,]?\s*the answer is|thus[,]?\s*the answer is|"
        r"答案是|最终答案是|正确答案是|选择)\s*[:：]?\s*\(?([ABCD])\)?\b"
    ),
    re.compile(r"(?i)option\s*([ABCD])\b"),
]

LAST_LETTER_PATTERN = re.compile(r"\b([ABCD])\b")


def extract_answer_letter(text: str) -> Tuple[Optional[str], str]:
    """从模型输出中抽取最终答案字母 A/B/C/D。"""
    if not text:
        return None, "empty_text"

    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper(), pattern.pattern

    tail = text[-500:]
    matches = LAST_LETTER_PATTERN.findall(tail)
    if matches:
        return matches[-1].upper(), "last_letter_fallback"

    return None, "not_found"


def pick_text_field(item: Dict, preferred_fields: List[str]) -> Tuple[str, str]:
    for field in preferred_fields:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value, field
    return "", ""


def read_json_from_zip(zf: zipfile.ZipFile, folder_idx: int) -> Tuple[Dict, str]:
    prefix = f"{folder_idx}/"
    candidates = [
        name for name in zf.namelist()
        if name.startswith(prefix) and name.endswith(".json") and not name.endswith("/")
    ]
    if not candidates:
        raise FileNotFoundError(f"压缩包中未找到 {folder_idx}/ 下的 json 文件")

    candidates = sorted(candidates, key=lambda x: (os.path.basename(x) != "0.json", x))
    json_path = candidates[0]
    data = json.loads(zf.read(json_path))

    if isinstance(data, list):
        if not data:
            raise ValueError(f"{json_path} 是空列表")
        item = data[-1]
    elif isinstance(data, dict):
        item = data
    else:
        raise TypeError(f"{json_path} 的 JSON 顶层既不是 list 也不是 dict")

    return item, json_path


def read_json_from_dir(root_dir: str, folder_idx: int) -> Tuple[Dict, str]:
    folder = os.path.join(root_dir, str(folder_idx))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"目录不存在：{folder}")

    candidates = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".json") and os.path.isfile(os.path.join(folder, f))
    ]
    if not candidates:
        raise FileNotFoundError(f"{folder} 下未找到 json 文件")

    candidates = sorted(candidates, key=lambda x: (os.path.basename(x) != "0.json", x))
    json_path = candidates[0]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not data:
            raise ValueError(f"{json_path} 是空列表")
        item = data[-1]
    elif isinstance(data, dict):
        item = data
    else:
        raise TypeError(f"{json_path} 的 JSON 顶层既不是 list 也不是 dict")

    return item, json_path


def get_available_indices_from_zip(zf: zipfile.ZipFile) -> List[int]:
    indices = set()
    for name in zf.namelist():
        parts = name.split("/")
        if parts and parts[0].isdigit():
            indices.add(int(parts[0]))
    return sorted(indices)


def get_available_indices_from_dir(root_dir: str) -> List[int]:
    indices = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.isdigit():
            indices.append(int(name))
    return sorted(indices)


def build_summary_text(
    total_dataset: int,
    total_evaluated: int,
    total_correct: int,
    accuracy: float,
    missing_indices: List[int],
    errors: List[Tuple[int, str, str]],
) -> str:
    lines = [
        "=" * 80,
        f"数据集总题数           : {total_dataset}",
        f"成功评估题数           : {total_evaluated}",
        f"正确题数               : {total_correct}",
        f"准确率（评估子集）     : {accuracy:.4%}",
        f"缺失题目数（未生成结果）: {len(missing_indices)}",
        f"异常题目数             : {len(errors)}",
        "=" * 80,
    ]

    if errors:
        lines.append("")
        lines.append("前 20 个异常示例：")
        for idx, etype, msg in errors[:20]:
            lines.append(f"  - index={idx} | {etype}: {msg}")

    if missing_indices:
        preview = missing_indices[:30]
        suffix = " ..." if len(missing_indices) > 30 else ""
        lines.append("")
        lines.append(f"缺失题号（前 30 个）: {preview}{suffix}")

    return "\n".join(lines)


def evaluate(
    result_path: str,
    parquet_path: str,
    output_csv: Optional[str],
    summary_txt: Optional[str],
    preferred_fields: List[str],
) -> None:
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise SystemExit(
            "读取 parquet 失败。请确认已安装 pyarrow：pip install pyarrow"
        ) from e

    required_cols = {"answer"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"parquet 缺少必要字段：{required_cols - set(df.columns)}")

    results = []
    errors = []

    if os.path.isfile(result_path) and result_path.endswith(".zip"):
        with zipfile.ZipFile(result_path, "r") as zf:
            indices = get_available_indices_from_zip(zf)
            for idx in indices:
                if idx >= len(df):
                    errors.append((idx, "index_out_of_range", "结果索引超过 parquet 行数"))
                    continue
                try:
                    item, json_path = read_json_from_zip(zf, idx)
                    text, used_field = pick_text_field(item, preferred_fields)
                    pred, method = extract_answer_letter(text)
                    gold = str(df.iloc[idx]["answer"]).strip().upper()
                    correct = (pred == gold)
                    results.append({
                        "index": idx,
                        "json_path": json_path,
                        "used_field": used_field,
                        "extract_method": method,
                        "pred": pred,
                        "gold": gold,
                        "correct": correct,
                    })
                except Exception as e:
                    errors.append((idx, type(e).__name__, str(e)))
    elif os.path.isdir(result_path):
        indices = get_available_indices_from_dir(result_path)
        for idx in indices:
            if idx >= len(df):
                errors.append((idx, "index_out_of_range", "结果索引超过 parquet 行数"))
                continue
            try:
                item, json_path = read_json_from_dir(result_path, idx)
                text, used_field = pick_text_field(item, preferred_fields)
                pred, method = extract_answer_letter(text)
                gold = str(df.iloc[idx]["answer"]).strip().upper()
                correct = (pred == gold)
                results.append({
                    "index": idx,
                    "json_path": json_path,
                    "used_field": used_field,
                    "extract_method": method,
                    "pred": pred,
                    "gold": gold,
                    "correct": correct,
                })
            except Exception as e:
                errors.append((idx, type(e).__name__, str(e)))
    else:
        raise SystemExit(f"结果路径不存在，或既不是目录也不是 zip：{result_path}")

    evaluated_indices = {r["index"] for r in results}
    all_indices = set(range(len(df)))
    missing_indices = sorted(all_indices - evaluated_indices)

    total_dataset = len(df)
    total_evaluated = len(results)
    total_correct = sum(int(r["correct"]) for r in results)
    accuracy = (total_correct / total_evaluated) if total_evaluated > 0 else 0.0

    summary_text = build_summary_text(
        total_dataset=total_dataset,
        total_evaluated=total_evaluated,
        total_correct=total_correct,
        accuracy=accuracy,
        missing_indices=missing_indices,
        errors=errors,
    )
    print(summary_text)

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index", "json_path", "used_field", "extract_method",
                    "pred", "gold", "correct"
                ],
            )
            writer.writeheader()
            writer.writerows(sorted(results, key=lambda x: x["index"]))
        print(f"详细结果已保存到: {output_csv}")

    if summary_txt:
        os.makedirs(os.path.dirname(os.path.abspath(summary_txt)), exist_ok=True)
        with open(summary_txt, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        print(f"统计摘要已保存到: {summary_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 GPQA 生成结果准确率")
    parser.add_argument(
        "--result_path",
        type=str,
        # required=True,
        default="router_result/gpqa",
        help="模型生成结果路径，支持目录或 zip 文件",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        # required=True,
        default="data/gpqa.parquet",
        help="GPQA 数据集 parquet 文件路径",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="router_result/gpqa/gpqa_eval_details.csv",
        help="保存逐题评估详情的 CSV 路径",
    )
    parser.add_argument(
        "--summary_txt",
        type=str,
        default="router_result/gpqa/gpqa_eval_summary.txt",
        help="保存统计摘要的 TXT 路径",
    )
    parser.add_argument(
        "--fields",
        type=str,
        default="base_model_step,step_str,small_model_step",
        help="按优先级尝试抽取答案的字段，逗号分隔",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preferred_fields = [x.strip() for x in args.fields.split(",") if x.strip()]
    evaluate(
        result_path=args.result_path,
        parquet_path=args.parquet_path,
        output_csv=args.output_csv,
        summary_txt=args.summary_txt,
        preferred_fields=preferred_fields,
    )
