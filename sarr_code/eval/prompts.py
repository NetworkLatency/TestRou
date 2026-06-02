from __future__ import annotations


def build_problem_text(problem: dict, dataset_name: str) -> str:
    if dataset_name in {"aime24", "aime25", "math500"}:
        body = problem.get("problem") or problem.get("question") or problem.get("prompt")
        return (
            "Solve the following math problem and return ONLY the final answer.\n"
            "Please reason step by step, separate logical reasoning steps with two newline characters "
            "(\\n\\n), and put your final answer within \\boxed{}.\n\n"
            f"Problem: {body}\n\n"
        )
    if dataset_name in {"gpqa", "gpqa_diamond"}:
        body = problem.get("problem") or problem.get("Question") or problem.get("question")
        choices = []
        for label in ["A", "B", "C", "D"]:
            value = problem.get(label) or problem.get(f"choice_{label.lower()}") or problem.get(f"Choice {label}")
            if value:
                choices.append(f"{label}. {value}")
        if not choices and problem.get("Correct Answer"):
            gpqa_values = [
                problem.get("Correct Answer"),
                problem.get("Incorrect Answer 1"),
                problem.get("Incorrect Answer 2"),
                problem.get("Incorrect Answer 3"),
            ]
            choices = [f"{label}. {value}" for label, value in zip(["A", "B", "C", "D"], gpqa_values) if value]
        choices_text = "\n".join(choices)
        if choices_text:
            body = f"{body}\n\n{choices_text}"
        return (
            "What is the correct answer to the following problem? Please reason step by step.\n"
            "Separate logical reasoning steps with two newline characters (\\n\\n).\n"
            "Put the final answer strictly in the format \\boxed{X}, where X is a single letter "
            "(A, B, C, or D).\n\n"
            f"Problem: {body}\n\n"
        )
    if dataset_name == "humaneval":
        prompt = problem.get("prompt") or problem.get("problem") or ""
        return (
            "Write Python code to solve the following programming task. "
            "HumanEval execution evaluation is not implemented for this SARR runner.\n\n"
            f"{prompt}"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")
