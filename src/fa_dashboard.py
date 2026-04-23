import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_logs(logs_path):
    root = Path(logs_path)
    step_logs = []
    problem_logs = []

    if root.is_file():
        if root.name.endswith(".steps.jsonl") or root.suffix == ".jsonl":
            step_logs.extend(_read_jsonl(root))
        elif root.name.endswith(".problem.json"):
            with open(root, "r", encoding="utf-8") as f:
                problem_logs.append(json.load(f))
    else:
        for path in root.rglob("*.steps.jsonl"):
            step_logs.extend(_read_jsonl(path))
        for path in root.rglob("*.problem.json"):
            with open(path, "r", encoding="utf-8") as f:
                problem_logs.append(json.load(f))

    return step_logs, problem_logs


def _values(rows, key):
    vals = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            vals.append(float(value))
    return vals


def _hist(ax, values, label, bins=40, alpha=0.45):
    if values:
        ax.hist(values, bins=bins, density=True, alpha=alpha, label=label)


def _ks_pvalue(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(a, b).pvalue)
    except Exception:
        return None


def _handoff_streaks(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("is_final_answer"):
            continue
        grouped[(row.get("problem_id"), row.get("repeat_id"))].append(row)

    streaks = []
    for _, items in grouped.items():
        items = sorted(items, key=lambda row: row.get("step_idx", 0))
        current = 0
        for row in items:
            is_handoff = row.get("actual_decision") == "HANDOFF" or row.get("generator") == "LLM"
            if is_handoff:
                current += 1
            elif current:
                streaks.append(current)
                current = 0
        if current:
            streaks.append(current)
    return streaks


def _transition_matrix(rows):
    labels = ["SLM", "LLM"]
    idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((2, 2), dtype=int)

    for row in rows:
        if row.get("is_final_answer"):
            continue
        prev = row.get("prev_step_provenance")
        curr = row.get("generator")
        if prev in idx and curr in idx:
            matrix[idx[prev], idx[curr]] += 1
    return labels, matrix


def _plot_pareto(ax, problem_logs, small_accuracy=None, large_accuracy=None):
    usable = [row for row in problem_logs if row.get("is_correct") is not None]
    if not usable:
        ax.text(
            0.5,
            0.5,
            "Accuracy unavailable\nPopulate problem_log['is_correct'] after evaluation.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    grouped = defaultdict(list)
    for row in usable:
        grouped[(row.get("routing_mode", "unknown"), row.get("tau"))].append(row)

    by_mode = defaultdict(list)
    for (mode, tau), rows in grouped.items():
        total_slm = sum(row.get("total_slm_tokens", 0) for row in rows)
        total_llm = sum(row.get("total_llm_tokens", 0) for row in rows)
        denom = total_slm + total_llm
        llm_share = total_llm / denom if denom else 0.0
        accuracy = sum(1 for row in rows if row.get("is_correct")) / len(rows)
        by_mode[mode].append((llm_share, accuracy, tau))

    for mode, points in by_mode.items():
        points = sorted(points)
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=mode)
        for llm_share, accuracy, tau in points:
            ax.annotate(str(tau), (llm_share, accuracy), fontsize=8)

    if small_accuracy is not None:
        ax.axhline(float(small_accuracy), color="gray", linestyle="--", linewidth=1, label="A_S")
    if large_accuracy is not None:
        ax.axhline(float(large_accuracy), color="black", linestyle="--", linewidth=1, label="A_L")
    if small_accuracy is not None and large_accuracy is not None:
        low, high = sorted([float(small_accuracy), float(large_accuracy)])
        ax.axhspan(low, high, color="gray", alpha=0.12)

    ax.set_xlabel("LLM token share")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pareto frontier")
    ax.legend(fontsize=8)


def generate_dashboard(logs_path, save_to, small_accuracy=None, large_accuracy=None):
    step_logs, problem_logs = _load_logs(logs_path)
    routed_steps = [row for row in step_logs if not row.get("is_final_answer")]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()

    raw = _values(routed_steps, "raw_H_init")
    content = _values(routed_steps, "H_content_init")
    _hist(axes[0], raw, "raw_H_init")
    _hist(axes[0], content, "H_content_init")
    axes[0].set_title("Entropy distribution")
    axes[0].set_xlabel("Entropy")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    skipped = [int(row.get("n_format_skipped", 0)) for row in routed_steps]
    if skipped:
        bins = np.arange(max(skipped) + 2) - 0.5
        axes[1].hist(skipped, bins=bins, rwidth=0.8)
    axes[1].set_title("Format tokens skipped per step")
    axes[1].set_xlabel("n_format_skipped")
    axes[1].set_ylabel("Count")

    token_counter = Counter()
    for row in routed_steps:
        token_counter.update(row.get("skipped_token_strs", []))
    top_tokens = token_counter.most_common(10)
    if top_tokens:
        labels, counts = zip(*top_tokens)
        axes[2].barh(range(len(labels)), counts)
        axes[2].set_yticks(range(len(labels)), labels)
        axes[2].invert_yaxis()
    axes[2].set_title("Most skipped format tokens")
    axes[2].set_xlabel("Count")

    raw_prev_slm = [float(row["raw_H_init"]) for row in routed_steps if row.get("prev_step_provenance") == "SLM" and row.get("raw_H_init") is not None]
    raw_prev_llm = [float(row["raw_H_init"]) for row in routed_steps if row.get("prev_step_provenance") == "LLM" and row.get("raw_H_init") is not None]
    content_prev_slm = [float(row["H_content_init"]) for row in routed_steps if row.get("prev_step_provenance") == "SLM" and row.get("H_content_init") is not None]
    content_prev_llm = [float(row["H_content_init"]) for row in routed_steps if row.get("prev_step_provenance") == "LLM" and row.get("H_content_init") is not None]
    _hist(axes[3], raw_prev_slm, "raw | prev=SLM", alpha=0.32)
    _hist(axes[3], raw_prev_llm, "raw | prev=LLM", alpha=0.32)
    _hist(axes[3], content_prev_slm, "content | prev=SLM", alpha=0.32)
    _hist(axes[3], content_prev_llm, "content | prev=LLM", alpha=0.32)
    raw_p = _ks_pvalue(raw_prev_slm, raw_prev_llm)
    content_p = _ks_pvalue(content_prev_slm, content_prev_llm)
    title = "Entropy by previous provenance"
    if raw_p is not None:
        title += f"\nKS raw p={raw_p:.2g}"
    if content_p is not None:
        title += f", content p={content_p:.2g}"
    axes[3].set_title(title)
    axes[3].set_xlabel("Entropy")
    axes[3].set_ylabel("Density")
    axes[3].legend(fontsize=7)

    _plot_pareto(axes[4], problem_logs, small_accuracy=small_accuracy, large_accuracy=large_accuracy)

    streaks = _handoff_streaks(routed_steps)
    if streaks:
        bins = np.arange(max(streaks) + 2) - 0.5
        axes[5].hist(streaks, bins=bins, rwidth=0.8)
    axes[5].set_title("Consecutive handoff streak length")
    axes[5].set_xlabel("Streak length")
    axes[5].set_ylabel("Count")

    labels, matrix = _transition_matrix(routed_steps)
    im = axes[6].imshow(matrix, cmap="Blues")
    axes[6].set_xticks(range(len(labels)), labels)
    axes[6].set_yticks(range(len(labels)), labels)
    axes[6].set_xlabel("Next generator")
    axes[6].set_ylabel("Previous generator")
    axes[6].set_title("Transition matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[6].text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    summary = [
        f"Step logs: {len(step_logs)}",
        f"Routed steps: {len(routed_steps)}",
        f"Problem logs: {len(problem_logs)}",
        f"Mean raw_H_init: {np.mean(raw):.4f}" if raw else "Mean raw_H_init: n/a",
        f"Mean H_content_init: {np.mean(content):.4f}" if content else "Mean H_content_init: n/a",
        f"Mean skipped: {np.mean(skipped):.4f}" if skipped else "Mean skipped: n/a",
    ]
    axes[7].text(0.02, 0.98, "\n".join(summary), va="top", transform=axes[7].transAxes)
    axes[7].set_axis_off()

    fig.tight_layout()
    save_path = Path(save_to)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return str(save_path)


def main():
    parser = argparse.ArgumentParser(description="Generate FA-Routing diagnostic dashboard.")
    parser.add_argument("--logs_path", help="Output directory or a .steps.jsonl/.problem.json file",default="router_result/aime25")
    parser.add_argument("--save_to", default="fa_dashboard.png", help="Dashboard image path")
    parser.add_argument("--small_accuracy", type=float, default=None, help="Optional A_S horizontal line")
    parser.add_argument("--large_accuracy", type=float, default=None, help="Optional A_L horizontal line")
    args = parser.parse_args()

    path = generate_dashboard(
        args.logs_path,
        args.save_to,
        small_accuracy=args.small_accuracy,
        large_accuracy=args.large_accuracy,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
