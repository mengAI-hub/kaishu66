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


def choose_id_fallback(text, choose):
    if not isinstance(choose, dict) or not choose:
        return "A"

    best_id = "A"
    best_score = -1

    for cid, option_text in choose.items():
        score = overlap_score(text, option_text)
        if score > best_score:
            best_score = score
            best_id = str(cid).upper()

    return best_id


def is_incomplete(item):
    if not str(item.get("choose_id", "")).strip():
        return True

    ans_qa_words = item.get("ans_qa_words", {})
    ans_qa_sents = item.get("ans_qa_sents", {})

    if not isinstance(ans_qa_words, dict):
        return True

    if not isinstance(ans_qa_sents, dict):
        return True

    for _, v in ans_qa_words.items():
        if not str(v).strip():
            return True

    for _, v in ans_qa_sents.items():
        if not str(v).strip():
            return True

    return False


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

    valid_ids = set(str(k).upper() for k in choose.keys()) if isinstance(choose, dict) else {"A", "B", "C", "D"}
    if choose_id not in valid_ids:
        emotion_text = str(obj.get("emotion", "")) + " " + str(obj.get("choose_text", ""))
        choose_id = choose_id_fallback(emotion_text, choose)

    return {
        "idx": row.get("idx"),
        "ans_qa_words": fixed_words,
        "ans_qa_sents": fixed_sents,
        "choose_id": choose_id
    }


def build_repair_prompt(row, old_item):
    qa_words = row.get("qa_words", [])
    qa_sents = row.get("qa_sents", [])
    choose = row.get("choose", {})

    empty_words = []
    for w in qa_words:
        w = str(w)
        if not str(old_item.get("ans_qa_words", {}).get(w, "")).strip():
            empty_words.append(w)

    empty_sents = []
    for s in qa_sents:
        s = str(s)
        if not str(old_item.get("ans_qa_sents", {}).get(s, "")).strip():
            empty_sents.append(s)

    choose_missing = not str(old_item.get("choose_id", "")).strip()

    return f"""请修复下面古诗词理解任务的输出。只输出严格 JSON。

【诗题】
{row.get("title", "")}

【作者】
{row.get("author", "")}

【诗文】
{row.get("content", "")}

【需要解释的词语 qa_words】
{json.dumps(qa_words, ensure_ascii=False)}

【需要翻译的诗句 qa_sents】
{json.dumps(qa_sents, ensure_ascii=False)}

【情感选项 choose】
{json.dumps(choose, ensure_ascii=False)}

【上一次输出存在的问题】
- 解释为空的词语：{json.dumps(empty_words, ensure_ascii=False)}
- 翻译为空的句子：{json.dumps(empty_sents, ensure_ascii=False)}
- choose_id 是否为空或非法：{"是" if choose_missing else "否"}

【严格要求】
1. ans_qa_words 必须解释 qa_words 中每一个词，key 必须和 qa_words 完全一致，不能漏。
2. ans_qa_sents 必须翻译 qa_sents 中每一句，key 必须和 qa_sents 完全一致，不能漏。
3. choose_id 必须从 choose 的 A/B/C/D 中选择一个。
4. 不能输出 Markdown，不能输出解释过程。
5. 只输出严格 JSON。

输出格式：
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
    parser.add_argument("--raw", required=True)
    parser.add_argument("--pred", required=True)
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

    raw_data = read_json(args.raw)
    pred_data = read_json(args.pred)

    if len(raw_data) != len(pred_data):
        raise ValueError(f"长度不一致：raw={len(raw_data)}, pred={len(pred_data)}")

    repaired = []
    target_count = 0
    success_count = 0

    for row, old_item in tqdm(list(zip(raw_data, pred_data))):
        if not is_incomplete(old_item):
            repaired.append(old_item)
            continue

        target_count += 1

        prompt = build_repair_prompt(row, old_item)

        raw = call_qwen_api(
            client=client,
            model=args.model,
            system="你是古诗词理解任务的 JSON 修复助手。你必须补全缺失字段，只输出严格 JSON。",
            user=prompt,
            temperature=args.temperature,
        )

        obj = extract_json(raw)
        new_item = normalize_output(obj, row)

        if not is_incomplete(new_item):
            success_count += 1
        else:
            print(f"\nWarning: idx={row.get('idx')} still incomplete")

        repaired.append(new_item)

    save_json(repaired, args.output)

    bad = [x for x in repaired if is_incomplete(x)]

    print(f"saved to {args.output}")
    print(f"repair targets: {target_count}")
    print(f"repair success: {success_count}")
    print(f"remaining incomplete: {len(bad)}")
    if bad:
        print("remaining bad idx:", [x.get("idx") for x in bad[:50]])


if __name__ == "__main__":
    main()