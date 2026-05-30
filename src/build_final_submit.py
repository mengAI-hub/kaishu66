import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def clean_task1(data):
    return [
        {
            "idx": item.get("idx"),
            "ans_qa_words": item.get("ans_qa_words", {}),
            "ans_qa_sents": item.get("ans_qa_sents", {}),
            "choose_id": str(item.get("choose_id", "")).strip().upper()
        }
        for item in data
    ]


def clean_task2(data):
    return [
        {
            "idx": item.get("idx"),
            "flag": item.get("flag", 0),
            "answer": item.get("answer", "")
        }
        for item in data
    ]


def clean_task3(data):
    out = []
    for item in data:
        ans = item.get("answer", [])
        if isinstance(ans, str):
            ans = [ans]
        out.append({
            "idx": item.get("idx"),
            "answer": ans
        })
    return out


def clean_task4(data):
    return [
        {
            "idx": item.get("idx"),
            "answer": str(item.get("answer", "")).strip().upper()
        }
        for item in data
    ]


def validate(submit):
    bad = []

    for k in ["task1", "task2", "task3", "task4"]:
        if k not in submit:
            bad.append((k, "missing"))
        elif not isinstance(submit[k], list):
            bad.append((k, "not list"))

    for item in submit.get("task1", []):
        if item.get("idx") is None:
            bad.append(("task1", "idx none"))
        if not isinstance(item.get("ans_qa_words"), dict):
            bad.append(("task1", item.get("idx"), "ans_qa_words not dict"))
        if not isinstance(item.get("ans_qa_sents"), dict):
            bad.append(("task1", item.get("idx"), "ans_qa_sents not dict"))
        if str(item.get("choose_id", "")).strip() == "":
            bad.append(("task1", item.get("idx"), "choose_id empty"))

    for item in submit.get("task2", []):
        if item.get("idx") is None:
            bad.append(("task2", "idx none"))
        if item.get("flag") not in [0, 1]:
            bad.append(("task2", item.get("idx"), "flag invalid"))

    for item in submit.get("task3", []):
        if item.get("idx") is None:
            bad.append(("task3", "idx none"))
        if not isinstance(item.get("answer"), list):
            bad.append(("task3", item.get("idx"), "answer not list"))

    for item in submit.get("task4", []):
        if item.get("idx") is None:
            bad.append(("task4", "idx none"))
        if item.get("answer") not in ["A", "B", "C", "D"]:
            bad.append(("task4", item.get("idx"), "answer invalid"))

    return bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1", required=True)
    parser.add_argument("--task2", required=True)
    parser.add_argument("--task3", required=True)
    parser.add_argument("--task4", required=True)
    parser.add_argument("--output", default="submit.json")
    args = parser.parse_args()

    submit = {
        "task1": clean_task1(read_json(args.task1)),
        "task2": clean_task2(read_json(args.task2)),
        "task3": clean_task3(read_json(args.task3)),
        "task4": clean_task4(read_json(args.task4)),
    }

    bad = validate(submit)
    save_json(submit, args.output)

    print(f"saved to {args.output}")
    print(f"validate bad count: {len(bad)}")
    if bad:
        print(bad[:30])


if __name__ == "__main__":
    main()