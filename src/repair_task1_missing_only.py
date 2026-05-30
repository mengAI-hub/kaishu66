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

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}


def normalize_key(s):
    s = str(s).strip()
    s = re.sub(r"[，。！？；：、“”‘’\"'\s\[\]（）()]", "", s)
    return s


def get_by_loose_key(d, target_key):
    if not isinstance(d, dict):
        return ""

    if target_key in d and str(d[target_key]).strip():
        return str(d[target_key]).strip()

    target_norm = normalize_key(target_key)

    for k, v in d.items():
        if normalize_key(k) == target_norm and str(v).strip():
            return str(v).strip()

    return ""


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


def get_missing(row, item):
    qa_words = [str(x) for x in row.get("qa_words", [])]
    qa_sents = [str(x) for x in row.get("qa_sents", [])]

    ans_words = item.get("ans_qa_words", {})
    ans_sents = item.get("ans_qa_sents", {})

    if not isinstance(ans_words, dict):
        ans_words = {}

    if not isinstance(ans_sents, dict):
        ans_sents = {}

    missing_words = []
    for w in qa_words:
        if not str(ans_words.get(w, "")).strip():
            missing_words.append(w)

    missing_sents = []
    for s in qa_sents:
        if not str(ans_sents.get(s, "")).strip():
            missing_sents.append(s)

    choose = row.get("choose", {})
    choose_id = str(item.get("choose_id", "")).strip().upper()
    valid_ids = set(str(k).upper() for k in choose.keys()) if isinstance(choose, dict) else {"A", "B", "C", "D"}

    choose_bad = choose_id not in valid_ids

    return missing_words, missing_sents, choose_bad


def is_incomplete(row, item):
    missing_words, missing_sents, choose_bad = get_missing(row, item)
    return bool(missing_words or missing_sents or choose_bad)


def build_prompt(row, item):
    missing_words, missing_sents, choose_bad = get_missing(row, item)

    return f"""请只补全下面古诗词理解任务中缺失的字段，并严格输出 JSON。

【诗题】
{row.get("title", "")}

【作者】
{row.get("author", "")}

【诗文】
{row.get("content", "")}

【只需要补充解释的词语】
{json.dumps(missing_words, ensure_ascii=False)}

【只需要补充翻译的诗句】
{json.dumps(missing_sents, ensure_ascii=False)}

【情感选项】
{json.dumps(row.get("choose", {}), ensure_ascii=False)}

【当前已有答案】
{json.dumps(item, ensure_ascii=False)}

要求：
1. 只需要回答缺失的词语和句子，不要重写已有答案。
2. ans_qa_words 只包含“只需要补充解释的词语”中的 key，key 必须完全一致。
3. ans_qa_sents 只包含“只需要补充翻译的诗句”中的 key，key 必须完全一致。
4. 如果 choose_id 已缺失或非法，请从 A/B/C/D 中选择一个最符合全诗情感的选项。
5. 如果某一类没有缺失，就输出空对象。
6. 只输出严格 JSON，不要 Markdown，不要解释过程。

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
        new_item = dict(old_item)

        if not is_incomplete(row, new_item):
            repaired.append(new_item)
            continue

        target_count += 1

        prompt = build_prompt(row, new_item)

        raw = call_qwen_api(
            client=client,
            model=args.model,
            system="你是古诗词理解任务的缺失字段修复助手。你只补缺失字段，只输出严格 JSON。",
            user=prompt,
            temperature=args.temperature,
        )

        obj = extract_json(raw)

        # 保留旧答案，只补空字段
        ans_words = dict(new_item.get("ans_qa_words", {}))
        ans_sents = dict(new_item.get("ans_qa_sents", {}))

        obj_words = obj.get("ans_qa_words", {})
        obj_sents = obj.get("ans_qa_sents", {})

        missing_words, missing_sents, choose_bad = get_missing(row, new_item)

        for w in missing_words:
            val = get_by_loose_key(obj_words, w)
            if val:
                ans_words[w] = val

        for s in missing_sents:
            val = get_by_loose_key(obj_sents, s)
            if val:
                ans_sents[s] = val

        new_item["ans_qa_words"] = ans_words
        new_item["ans_qa_sents"] = ans_sents

        if choose_bad:
            choose_id = str(obj.get("choose_id", "")).strip().upper()
            choose = row.get("choose", {})
            valid_ids = set(str(k).upper() for k in choose.keys()) if isinstance(choose, dict) else {"A", "B", "C", "D"}

            if choose_id in valid_ids:
                new_item["choose_id"] = choose_id
            else:
                new_item["choose_id"] = choose_id_fallback(str(obj), choose)

        if not is_incomplete(row, new_item):
            success_count += 1
        else:
            mw, ms, cb = get_missing(row, new_item)
            print(f"\nWarning idx={row.get('idx')} still incomplete; words={mw}, sents={ms}, choose_bad={cb}")

        repaired.append(new_item)

    save_json(repaired, args.output)

    bad = [item for row, item in zip(raw_data, repaired) if is_incomplete(row, item)]

    print(f"saved to {args.output}")
    print(f"missing-only repair targets: {target_count}")
    print(f"missing-only repair success: {success_count}")
    print(f"remaining incomplete: {len(bad)}")
    if bad:
        print("remaining bad idx:", [x.get("idx") for x in bad[:50]])


if __name__ == "__main__":
    main()