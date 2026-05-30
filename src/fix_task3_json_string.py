import argparse
import json
import re
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def try_fix_answer(ans):
    if not isinstance(ans, list):
        return ans

    if len(ans) != 1:
        return ans

    s = str(ans[0]).strip()

    if '"answer"' not in s and not s.startswith("{"):
        return ans

    # 尝试正常 JSON
    try:
        obj = json.loads(s)
        fixed = obj.get("answer", ans)
        if isinstance(fixed, list):
            return [str(x).strip() for x in fixed if str(x).strip()]
    except Exception:
        pass

    # 尝试从字符串里抽取 ["...", "..."]
    m = re.search(r'"answer"\s*:\s*\[([\s\S]*?)\]', s)
    if m:
        inner = m.group(1)
        parts = re.findall(r'"([^"]+)"', inner)
        if parts:
            return [x.strip() for x in parts if x.strip()]

    # 再兜底：单引号内容
    parts = re.findall(r"'([^']+)'", s)
    if parts:
        return [x.strip() for x in parts if x.strip()]

    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = read_json(args.input)

    changed = []

    for item in data:
        old = item.get("answer", [])
        new = try_fix_answer(old)
        if new != old:
            item["answer"] = new
            changed.append(item.get("idx"))

    save_json(data, args.output)

    print(f"saved to {args.output}")
    print("changed idx:", changed)


if __name__ == "__main__":
    main()
