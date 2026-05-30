import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def extract_answer_list(text):
    """
    从模型输出中提取 answer 数组。
    期望格式：
    {"answer": ["第一空", "第二空"]}
    """
    text = (text or "").strip()

    # 第一种：模型直接输出标准 JSON
    try:
        obj = json.loads(text)
        ans = obj.get("answer", [])
        if isinstance(ans, list):
            return [str(x).strip() for x in ans]
        if isinstance(ans, str):
            return [ans.strip()]
    except Exception:
        pass

    # 第二种：模型输出中夹杂了 JSON
    match = re.search(r"\{[\s\S]*?\}", text)
    if match:
        try:
            obj = json.loads(match.group(0))
            ans = obj.get("answer", [])
            if isinstance(ans, list):
                return [str(x).strip() for x in ans]
            if isinstance(ans, str):
                return [ans.strip()]
        except Exception:
            pass

    # 第三种：兜底处理，如果模型输出多行，就按行切分
    lines = [
        x.strip(" 　，。；;：:\"'[]")
        for x in text.splitlines()
        if x.strip()
    ]

    if lines:
        return lines

    return [""]


def normalize_text(s):
    """
    用于宽松比较：
    去掉常见标点、空格。
    """
    s = str(s).strip()
    s = re.sub(r"[，。！？；：、“”‘’\"'\s\[\]（）()]", "", s)
    return s


def answers_equal(pred, gold):
    """
    训练集评估用。
    只做简单严格/半严格匹配。
    """
    if not isinstance(gold, list):
        gold = [gold]

    pred = [normalize_text(x) for x in pred]
    gold = [normalize_text(x) for x in gold]

    return pred == gold


def fill_template(template, row):
    return template.format(
        que=row.get("que", "")
    )


def call_qwen_api(client, model, system, user, temperature=0.2, max_retries=3):
    """
    调用通义千问 / DashScope OpenAI 兼容接口。

    注意：
    qwen3 系列在非流式调用时，需要显式设置：
    extra_body={"enable_thinking": False}
    否则会报：
    parameter.enable_thinking must be set to false for non-streaming calls
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    },
                ],
                "temperature": temperature,
                "top_p": 0.9,
            }

            # qwen3 非流式调用需要关闭 thinking
            if model.startswith("qwen3"):
                kwargs["extra_body"] = {
                    "enable_thinking": False
                }

            response = client.chat.completions.create(**kwargs)

            return response.choices[0].message.content

        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"API call failed after {max_retries} retries: {last_error}"
    )


def vote_answer_lists(preds):
    """
    对答案数组进行投票。
    完全相同的答案数组得分累加。

    preds 示例：
    [
      {"answer": ["郑既知亡矣", "若亡郑而有益于君"], "weight": 1.0},
      {"answer": ["郑既知亡矣", "若亡郑而有益于君"], "weight": 0.9}
    ]
    """
    score = defaultdict(float)
    original = {}

    for pred in preds:
        ans_list = pred["answer"]

        key = json.dumps(
            [normalize_text(x) for x in ans_list],
            ensure_ascii=False
        )

        score[key] += float(pred.get("weight", 1.0))
        original[key] = ans_list

    best_key = sorted(
        score.items(),
        key=lambda x: x[1],
        reverse=True
    )[0][0]

    return original[best_key]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--model", default="qwen2.5-7b-instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
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

    data = read_json(args.test)
    prompts = read_json(args.prompts)

    outputs = []
    total = 0
    correct = 0
    has_gold = False

    for i, row in enumerate(tqdm(data), start=1):
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
                    system=prompt.get("system", "你是古诗文补全专家。"),
                    user=user_prompt,
                    temperature=args.temperature,
                )

                ans_list = extract_answer_list(raw)

                preds.append({
                    "answer": ans_list,
                    "weight": float(prompt.get("weight", 1.0)),
                })

        final_answer = vote_answer_lists(preds)

        
        out_idx = row.get("idx")
        if out_idx is None:
           out_idx = i - 1

        item = {
            "idx": out_idx,
            "answer": final_answer
}

        if "answer" in row:
            has_gold = True
            gold = row["answer"]
            item["gold"] = gold
            item["correct"] = answers_equal(final_answer, gold)
            correct += int(item["correct"])

        outputs.append(item)
        total += 1

    save_json(outputs, args.output)

    print(f"saved to {args.output}")

    if has_gold:
        acc = correct / total if total else 0
        print(f"accuracy: {correct}/{total} = {acc:.4f}")


if __name__ == "__main__":
    main()