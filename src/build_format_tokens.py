import argparse
import json
import re

from transformers import AutoTokenizer


FORMAT_STRINGS = [
    "*", "**", "***",
    "- ", "* ", "+ ",
    "-", "—",
    "#", "##", "###", "####",
    "\"", "'", "`", "```",
    "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.",
    "(1)", "(2)", "(3)",
    "\n", "\n\n", "\t",
    " ","(",'[', '|',"(-",'√','"'
]

MULTI_TOKEN_SAFE_STRIPPED = {"*", "**", "***", "-", "—", "#", "##", "###", "####", "\"", "'", "`", "```", ""}


def is_pure_format_piece(text):
    stripped = text.strip()
    if stripped in MULTI_TOKEN_SAFE_STRIPPED:
        return True
    return bool(re.fullmatch(r"\(?\d+\)\.?", stripped) or re.fullmatch(r"\d+\.", stripped))


def build_format_tokens(tokenizer_name_or_path):
    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    format_token_ids = set()

    for text in FORMAT_STRINGS:
        for prefix in ["", " "]:
            ids = tok.encode(prefix + text, add_special_tokens=False)
            if len(ids) == 1:
                format_token_ids.add(ids[0])
            else:
                for tid in ids:
                    decoded = tok.decode([tid])
                    if is_pure_format_piece(decoded):
                        format_token_ids.add(tid)

    decoded = {tid: tok.decode([tid]) for tid in sorted(format_token_ids)}
    return {
        "tokenizer": tokenizer_name_or_path,
        "ids": sorted(format_token_ids),
        "decoded": decoded,
        "format_strings": FORMAT_STRINGS,
        "manual_inspection_note": (
            "Inspect decoded before running FA modes. Remove any token that carries content, "
            "for example merged tokens like '**Step'."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Build a Qwen3 format-token whitelist for FA-Routing.")
    parser.add_argument("--tokenizer", default="/home/zhaoyang/Documents/code/models/Qwen3-4b/", help="Tokenizer name or local path")
    parser.add_argument("--output", default="data/format_tokens.json", help="Where to write the whitelist JSON")
    args = parser.parse_args()

    data = build_format_tokens(args.tokenizer)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Found {len(data['ids'])} format tokens")
    print(f"Wrote {args.output}")
    print("Please manually inspect the decoded field before using fa_skip or fa_strip.")


if __name__ == "__main__":
    main()
