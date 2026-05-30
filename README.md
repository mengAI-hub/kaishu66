# CCL2026 古诗词任务：仿 CCL25「风格改写 + LoRA + 加权投票」Starter Code

这套代码仿照你给的 CCL25-Eval 任务5系统报告思路，但把模型限制改为 **10B 以下**：

- 默认基座：`Qwen/Qwen2.5-7B-Instruct`
- 可替换为：`Qwen/Qwen3-8B`、`meta-llama/Llama-3.1-8B-Instruct` 等 10B 以下开源模型
- 不使用 RAG：测试阶段不检索外部知识库，不构建向量库
- 核心模块：
  1. 官方数据清洗与格式统一
  2. 可选：训练集风格改写 / 伪标注统一
  3. 可选：外部辅助训练集构建，但仅用于训练阶段
  4. LoRA / QLoRA 微调
  5. 多 prompt 多次生成 + 加权投票
  6. 严格 JSON 提交格式生成

## 对应论文思路如何迁移

原论文使用 Qwen2.5-14B-Instruct，本文代码默认改成 Qwen2.5-7B-Instruct，以符合 10B 以下限制。

| 论文思路 | 代码实现 |
|---|---|
| 数据清洗 | `src/clean_official_data.py` |
| 风格改写 | `src/rewrite_style.py` |
| 辅助训练集构建 | `src/build_aux_data.py` |
| 指令微调 / LoRA | `src/train_lora.py` |
| 多轮生成 | `src/infer_vote.py` |
| 加权投票 | `src/voting.py` |
| 提交格式校验 | `src/validate_submission.py` |

## 安装

```bash
pip install -r requirements.txt
```

## 第 1 步：放入官方数据

把官方训练集放到：

```bash
data/official_train.json
```

把官方验证集/测试集放到：

```bash
data/official_test.json
```

官方字段如果不是 `que/content/title/author/options/answer/flag`，可以在 `src/utils.py` 的 `normalize_row()` 里加字段映射。

## 第 2 步：清洗官方数据

```bash
python src/clean_official_data.py --input data/official_train.json --output data/official_train.clean.json
```

## 第 3 步：可选，构建辅助训练数据

如果你有外部诗词原文、注释、译文、典故解释，可以整理为 JSONL：

```json
{"task": "allusion", "que": "怀旧空吟闻笛赋，到乡翻似烂柯人。", "flag": 1, "answer": "闻笛赋、烂柯人均为典故..."}
{"task": "allusion", "que": "明月松间照，清泉石上流。", "flag": 0, "answer": ""}
```

然后运行：

```bash
python src/build_aux_data.py --input data/aux_poetry.jsonl --output data/aux_poetry.clean.jsonl
```

## 第 4 步：构造 SFT 数据

典故识别：

```bash
python src/build_sft_data.py --official data/official_train.clean.json --aux data/aux_poetry.clean.jsonl --task allusion --output data/sft_allusion.jsonl
```

情感分类：

```bash
python src/build_sft_data.py --official data/official_train.clean.json --task emotion --output data/sft_emotion.jsonl
```

## 第 5 步：LoRA / QLoRA 微调

```bash
python src/train_lora.py --config config.yaml
```

默认开启 4bit QLoRA。显存大可以把 `use_4bit: false`。

## 第 6 步：多 prompt + 加权投票推理

典故识别：

```bash
python src/infer_vote.py --model Qwen/Qwen2.5-7B-Instruct --adapter outputs/lora_poetry --test data/official_test.json --task allusion --prompts prompts/allusion_prompts.json --output outputs/submission_allusion.json
```

情感分类：

```bash
python src/infer_vote.py --model Qwen/Qwen2.5-7B-Instruct --adapter outputs/lora_poetry --test data/official_test.json --task emotion --prompts prompts/emotion_prompts.json --output outputs/submission_emotion.json
```

## 第 7 步：校验提交格式

```bash
python src/validate_submission.py --file outputs/submission_allusion.json --task allusion
```

## 建议实验顺序

1. `Qwen2.5-7B-Instruct` zero-shot + 多 prompt 投票
2. 官方 20 条 + 自构造 1000 条 LoRA
3. 加入外部训练样本，但注意风格一致
4. 比较是否真的涨分；如果不涨，删除外部数据

## 注意

这套代码的测试阶段没有 RAG，不会检索古诗词数据库。外部数据只在训练阶段使用。
