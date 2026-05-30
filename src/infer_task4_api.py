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


def options_to_text(options):
    if isinstance(options, dict):
        return "\n".join([f"{k}. {v}" for k, v in options.items()])

    if isinstance(options, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join([f"{letters[i]}. {v}" for i, v in enumerate(options)])

    return str(options)


def zhushi_to_text(zhushi):
    if not zhushi:
        return "无"

    if isinstance(zhushi, list):
        return "\n".join([str(x) for x in zhushi])

    return str(zhushi)


def extract_answer(text):
    """
    从模型输出中提取 A/B/C/D。
    期望模型输出：
    {"answer": "A"}
    """
    text = (text or "").strip()

    try:
        obj = json.loads(text)
        ans = str(obj.get("answer", "")).strip().upper()
        m = re.search(r"[A-D]", ans)
        if m:
            return m.group(0)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = str(obj.get("answer", "")).strip().upper()
            m2 = re.search(r"[A-D]", ans)
            if m2:
                return m2.group(0)
        except Exception:
            pass

    m = re.search(r"\b([A-D])\b", text.upper())
    if m:
        return m.group(1)

    return "A"


def get_question_text(question):
    """
    兼容训练集和测试集：
    训练集里可能叫 que
    测试集里叫 question
    """
    return question.get("que", question.get("question", ""))


def fill_template(template, poem, question):
    return template.format(
        title=poem.get("title", ""),
        author=poem.get("author", ""),
        content=poem.get("content", ""),
        zhushi_text=zhushi_to_text(poem.get("zhushi", [])),
        question=get_question_text(question),
        options_text=options_to_text(question.get("options", {})),
    )


def call_qwen_api(client, model, system, user, temperature=0.2, max_retries=3):
    last_error = None

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "top_p": 0.9,
            }

            # qwen3 系列非流式调用时需要关闭 thinking
            if model.startswith("qwen3"):
                kwargs["extra_body"] = {
                    "enable_thinking": False
                }

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


def vote_answers(preds):
    score = defaultdict(float)

    for pred in preds:
        ans = pred["answer"]
        weight = float(pred.get("weight", 1.0))
        score[ans] += weight

    return sorted(score.items(), key=lambda x: x[1], reverse=True)[0][0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", required=True)
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

    data = read_json(args.test)
    prompts = read_json(args.prompts)

    outputs = []
    total = 0
    correct = 0
    has_gold = False

    for poem_idx, poem in enumerate(tqdm(data), start=1):
        questions = poem.get("questions", [])

        for q_idx, question in enumerate(questions, start=1):
            preds = []

            for prompt in prompts:
                for _ in range(args.passes_per_prompt):
                    user_prompt = fill_template(
                        prompt["template"],
                        poem,
                        question
                    )

                    raw = call_qwen_api(
                        client=client,
                        model=args.model,
                        system=prompt.get("system", "你是古诗词理解专家。"),
                        user=user_prompt,
                        temperature=args.temperature,
                    )

                    ans = extract_answer(raw)

                    preds.append({
                        "answer": ans,
                        "weight": float(prompt.get("weight", 1.0)),
                    })

            final_answer = vote_answers(preds)

            # 测试集 poem 里有 idx，就优先使用 poem 的 idx。
            # 如果 question 里有 idx，则优先 question 的 idx。
            # 如果都没有，就用 total，从 0 开始。
            out_idx = question.get("idx", poem.get("idx"))

            if out_idx is None:
                out_idx = total

            item = {
                "idx": out_idx,
                "poem_idx": poem_idx,
                "question_idx": q_idx,
                "answer": final_answer
            }

            # 训练集有 gold 时，保留本地评估字段；测试集没有 answer，不会出现这些字段。
            if "answer" in question:
                has_gold = True
                gold = str(question["answer"]).strip().upper()
                item["gold"] = gold
                item["correct"] = final_answer == gold
                correct += int(final_answer == gold)

            outputs.append(item)
            total += 1

    save_json(outputs, args.output)

    print(f"saved to {args.output}")

    if has_gold:
        acc = correct / total if total else 0
        print(f"accuracy: {correct}/{total} = {acc:.4f}")


if __name__ == "__main__":
    main()