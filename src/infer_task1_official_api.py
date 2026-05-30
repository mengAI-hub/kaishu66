import argparse
import json
import os
import re
import time
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


def extract_json(text):
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}


def overlap_score(a, b):
    a = str(a)
    b = str(b)
    if not a or not b:
        return 0
    return len(set(a) & set(b))


def choose_id_fallback(emotion_text, choose):
    if not choose:
        return "A"

    if isinstance(choose, dict):
        best_id = "A"
        best_score = -1
        for cid, text in choose.items():
            score = overlap_score(emotion_text, text)
            if score > best_score:
                best_score = score
                best_id = str(cid)
        return best_id

    return "A"


def normalize_output(obj, row):
    qa_words = row.get("qa_words", [])
    qa_sents = row.get("qa_sents", [])
    choose = row.get("choose", {})

    ans_qa_words = obj.get("ans_qa_words", {})
    ans_qa_sents = obj.get("ans_qa_sents", {})
    choose_id = str(obj.get("choose_id", "")).strip().upper()

    if not isinstance(ans_qa_words, dict):
        ans_qa_words = {}

    if not isinstance(ans_qa_sents, dict):
        ans_qa_sents = {}

    fixed_words = {}
    for w in qa_words:
        w = str(w)
        fixed_words[w] = str(ans_qa_words.get(w, "")).strip()

    fixed_sents = {}
    for s in qa_sents:
        s = str(s)
        fixed_sents[s] = str(ans_qa_sents.get(s, "")).strip()

    if isinstance(choose, dict):
        valid_ids = set(str(k).upper() for k in choose.keys())
        if choose_id not in valid_ids:
            emotion_text = str(obj.get("emotion", ""))
            choose_id = choose_id_fallback(emotion_text, choose)
    else:
        if not choose_id:
            choose_id = "A"

    return {
        "idx": row.get("idx"),
        "ans_qa_words": fixed_words,
        "ans_qa_sents": fixed_sents,
        "choose_id": choose_id
    }


def is_incomplete(item):
    if not str(item.get("choose_id", "")).strip():
        return True

    for _, v in item.get("ans_qa_words", {}).items():
        if not str(v).strip():
            return True

    for _, v in item.get("ans_qa_sents", {}).items():
        if not str(v).strip():
            return True

    return False


def build_prompt(row):
    return f"""请完成古诗词理解任务，只输出严格 JSON。

【诗题】
{row.get("title", "")}

【作者】
{row.get("author", "")}

【诗文】
{row.get("content", "")}

【需要解释的词语 qa_words】
{json.dumps(row.get("qa_words", []), ensure_ascii=False)}

【需要翻译的诗句 qa_sents】
{json.dumps(row.get("qa_sents", []), ensure_ascii=False)}

【情感选项 choose】
{json.dumps(row.get("choose", {}), ensure_ascii=False)}

输出要求：
1. ans_qa_words：解释 qa_words 中每一个词，key 必须和 qa_words 完全一致。
2. ans_qa_sents：翻译 qa_sents 中每一个句子，key 必须和 qa_sents 完全一致。
3. choose_id：从 choose 的 A/B/C/D 中选择最符合诗歌情感的一项。
4. 不要输出 Markdown，不要解释过程。
5. 只输出严格 JSON。

格式：
{{
  "ans_qa_words": {{"词语": "解释"}},
  "ans_qa_sents": {{"诗句": "现代汉语翻译"}},
  "choose_id": "A"
}}"""


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

            if model.startswith("qwen3"):
                kwargs["extra_body"] = {"enable_thinking": False}

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as exc:
            last_error = exc
            time.sleep(2 ** attempt)

    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen2.5-7b-instruct")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--base-url", default=None)
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")

    base_url = args.base_url or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    client = OpenAI(api_key=api_key, base_url=base_url)

    data = read_json(args.test)
    outputs = []

    for row in tqdm(data):
        prompt = build_prompt(row)

        raw = call_qwen_api(
            client=client,
            model=args.model,
            system="你是古诗词理解专家。你必须严格按照 JSON 格式输出答案。",
            user=prompt,
            temperature=args.temperature,
        )

        obj = extract_json(raw)
        item = normalize_output(obj, row)
        outputs.append(item)

    save_json(outputs, args.output)

    bad = [x for x in outputs if is_incomplete(x)]
    print(f"saved to {args.output}")
    print(f"bad count: {len(bad)}")
    if bad:
        print("bad idx:", [x.get("idx") for x in bad[:30]])


if __name__ == "__main__":
    main()