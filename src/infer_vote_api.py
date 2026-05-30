import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def read_json_or_jsonl(path):
    text = Path(path).read_text(encoding="utf-8").strip()

    if path.endswith(".jsonl"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)

    if isinstance(data, list):
        return data

    for key in ["data", "test", "train", "examples", "items"]:
        if key in data and isinstance(data[key], list):
            return data[key]

    return [data]


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def normalize_row(row):
    """
    兼容不同官方字段。
    目前 task2.json 的格式是：
    {
      "que": "...",
      "answer": "..."
    }

    其中没有 idx，所以后面会用 i+1 自动补编号。
    """
    return {
        "idx": row.get("idx", row.get("id", row.get("index"))),
        "que": row.get(
            "que",
            row.get(
                "question",
                row.get(
                    "sentence",
                    row.get("text", "")
                )
            )
        ),
        "title": row.get("title", ""),
        "author": row.get("author", ""),
        "content": row.get(
            "content",
            row.get(
                "poem",
                row.get("poetry", "")
            )
        ),
        "options": row.get("options", row.get("choices", None)),
    }


def options_to_text(options):
    if options is None:
        return ""

    if isinstance(options, dict):
        return "\n".join([f"{k}. {v}" for k, v in options.items()])

    if isinstance(options, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join([f"{letters[i]}. {v}" for i, v in enumerate(options)])

    return str(options)


def extract_json(text):
    """
    尽量从模型输出中抽取 JSON。
    如果模型输出多余解释，也会尽量恢复。
    """
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    # 兜底：抽取 A/B/C/D
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return {"answer": match.group(1)}

    # 兜底：抽取 flag
    if re.search(r"flag[\"']?\s*[:：]\s*1", text):
        return {"flag": 1, "answer": ""}

    return {"flag": 0, "answer": ""}


def normalize_allusion_pred(obj):
    try:
        flag = 1 if int(obj.get("flag", 0)) == 1 else 0
    except Exception:
        flag = 0

    answer = str(obj.get("answer", "") or "").strip()

    if flag == 0:
        answer = ""

    return {
        "flag": flag,
        "answer": answer
    }


def normalize_emotion_pred(obj):
    ans = str(obj.get("answer", "") or "").strip().upper()
    match = re.search(r"[A-D]", ans)

    return {
        "answer": match.group(0) if match else "A"
    }


def vote_allusion(preds):
    """
    典故识别加权投票。
    preds 结构：
    [
      {"flag": 1, "answer": "...", "weight": 1.0},
      {"flag": 0, "answer": "", "weight": 0.75}
    ]
    """
    score = defaultdict(float)
    answers = []

    for pred in preds:
        flag = int(pred.get("flag", 0))
        weight = float(pred.get("weight", 1.0))

        score[flag] += weight

        if flag == 1 and pred.get("answer"):
            answers.append((weight, pred["answer"]))

    final_flag = 1 if score[1] > score[0] else 0

    if final_flag == 0:
        return {
            "flag": 0,
            "answer": ""
        }

    answers = sorted(
        answers,
        key=lambda x: (x[0], min(len(x[1]), 180)),
        reverse=True
    )

    return {
        "flag": 1,
        "answer": answers[0][1] if answers else ""
    }


def vote_emotion(preds):
    """
    情感分类加权投票。
    """
    score = defaultdict(float)

    for pred in preds:
        ans = str(pred.get("answer", "A")).strip().upper()[:1]

        if ans not in "ABCD":
            ans = "A"

        score[ans] += float(pred.get("weight", 1.0))

    best = sorted(score.items(), key=lambda x: x[1], reverse=True)[0][0]

    return {
        "answer": best
    }


def fill_template(template, row):
    return template.format(
        que=row.get("que", ""),
        title=row.get("title", ""),
        author=row.get("author", ""),
        content=row.get("content") or row.get("que", ""),
        options_text=options_to_text(row.get("options")),
    )


def call_qwen_api(client, model, system, user, temperature=0.2, max_retries=3):
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    },
                ],
                temperature=temperature,
                top_p=0.9,
            )

            return response.choices[0].message.content

        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"API call failed after {max_retries} retries: {last_error}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", required=True)
    parser.add_argument("--task", required=True, choices=["allusion", "emotion"])
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--model", default="qwen2.5-7b-instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--passes-per-prompt", type=int, default=1)
    parser.add_argument("--base-url", default=None)

    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")

    base_url = args.base_url or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    rows = [
        normalize_row(x)
        for x in read_json_or_jsonl(args.test)
    ]

    prompts = json.loads(
        Path(args.prompts).read_text(encoding="utf-8")
    )

    outputs = []

    for i, row in enumerate(tqdm(rows)):
        preds = []

        for prompt in prompts:
            for _ in range(args.passes_per_prompt):
                user_prompt = fill_template(
                    prompt["template"],
                    row
                )

                raw = call_qwen_api(
                    client=client,
                    model=args.model,
                    system=prompt.get("system", "你是古诗词理解专家。"),
                    user=user_prompt,
                    temperature=args.temperature,
                )

                obj = extract_json(raw)

                if args.task == "allusion":
                    pred = normalize_allusion_pred(obj)
                else:
                    pred = normalize_emotion_pred(obj)

                pred["weight"] = float(prompt.get("weight", 1.0))
                preds.append(pred)

        if args.task == "allusion":
            final = vote_allusion(preds)

            outputs.append({
                # 这里是关键修改：
                # 如果官方数据没有 idx，就自动用 i+1 生成编号
                "idx": row.get("idx") if row.get("idx") is not None else i,
                "flag": final["flag"],
                "answer": final["answer"],
            })

        else:
            final = vote_emotion(preds)

            outputs.append({
                # 这里也同样修复 idx=null 问题
                "idx": row.get("idx") if row.get("idx") is not None else i,
                "answer": final["answer"],
            })

    save_json(outputs, args.output)

    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()