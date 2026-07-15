# 新增评测用例指南

新增问题前先阅读 `sample_docs/`，不要根据系统当前回答反推标准答案。标准答案应来自知识库原文。

## 1. 确认段落编号

文档按空行切段，引用格式为：

```text
文件名#p段落序号
```

可以先运行一个问题查看候选和引用：

```bash
python3 -m rag_harness.cli ask \
  "生产发布的时间窗口是什么？" \
  --docs sample_docs
```

如果修改了文档段落结构，段落编号可能变化，对应的 `expected_citations` 也需要更新。

## 2. 正常回答用例

```json
{
  "id": "case_16",
  "category": "multi_paragraph",
  "question": "你的问题",
  "expected_sources": ["release_policy.md"],
  "expected_citations": [
    "release_policy.md#p2",
    "release_policy.md#p3"
  ],
  "keywords": ["关键词A", "关键词B"],
  "aspects": [
    {"name": "要点A", "keywords": ["关键词A"]},
    {"name": "要点B", "keywords": ["关键词B"]}
  ]
}
```

字段说明：

- `expected_sources`：文档级来源，便于人阅读；
- `expected_citations`：真正参与Precision/Recall计算的段落级标准；
- `keywords`：保留的整体关键词覆盖指标；
- `aspects`：复合问题的独立要点，每个要点的关键词必须全部出现才算覆盖；
- `category`：用于分类汇总，不影响检索逻辑。

## 3. 预期失败用例

```json
{
  "id": "case_17",
  "category": "expected_failure",
  "question": "知识库中不存在的问题",
  "expected_sources": [],
  "expected_citations": [],
  "keywords": [],
  "aspects": [],
  "expected_failure": "no_result"
}
```

预期失败不是坏结果。系统返回指定的 `failure_reason` 时，该用例的 `case_passed` 为 `true`。

## 4. 先检查JSON，再运行评测

```bash
python3 -m json.tool eval_cases.json > /dev/null
python3 -m rag_harness.cli eval \
  --cases eval_cases.json \
  --docs sample_docs \
  --output artifacts/eval_report.local.json \
  --trace-dir artifacts/traces \
  --fail-under 1.0
```

## 5. 失败时看什么

先在报告对应case中检查：

1. `missing_citations`：必要证据是否漏掉；
2. `unexpected_citations`：是否加入无关证据；
3. `aspects[].missing`：具体漏了哪个回答要点；
4. `failure_reason`：是否是 `no_result`、`timeout` 或 `tool_boundary`；
5. 对应JSONL中的 `evidence_selected.decisions`：候选是被分数、标题过滤还是Token预算拒绝。

不要立即降低标准答案或调阈值。先判断是：问题写得含糊、标准答案标错、知识库内容不足，还是检索/选择逻辑真的存在缺陷。

## 6. 有价值的新增用例

- 同一词出现在多个文档、但语义不同的困难负样本；
- 需要3段以上证据的多要点问题；
- 跨文档的流程问题；
- 明确无答案的问题；
- 参数边界、超时和越权工具；
- 改写表达、缩写和同义词问题。

新增用例后，最好先记录当前版本的失败结果，再修改代码并增加自动测试。这样可以区分“为了现有答案修改标准”与“用新证据修复真实缺陷”。
