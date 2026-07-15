# RAG Agent Harness

[![CI](https://github.com/Bowen-studying/rag-agent-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/Bowen-studying/rag-agent-harness/actions/workflows/ci.yml)

一个零外部依赖、可以在本地完整复现的 **RAG Agent 测试台**。它不只回答问题，还检查 Agent 是否找全证据、是否夹带错误引用、是否覆盖问题的每个要点，以及失败究竟发生在哪一步。

> 目标不是再做一个“看起来能回答”的 Demo，而是让一次 Agent 修改能够被测试、比较、诊断和安全地回归。

项目使用 Python 标准库，Python 3.10+；不需要 API Key、外部大模型、向量数据库或网络服务。

## 当前能验证什么

- 5 份 NovaLab 虚构制度文档，按段落建立可引用索引；
- 15 条固定评测：单证据、多段证据、跨文档、困难负样本和预期失败；
- 引用 Precision / Recall / F1、精确引用匹配率和问题要点覆盖率；
- BM25 检索、复合问题分解、动态证据数量和 Token 预算；
- `no_result`、超时、参数错误、工具越权和 Checkpoint 续跑；
- JSONL 轨迹记录候选证据、选择原因、拒绝原因和失败类型；
- 16 项自动测试与 GitHub Actions 持续验证。

## 使用了什么例子

仓库内置完全虚构的公司 **NovaLab**，示例知识库位于 `sample_docs/`：

| 文档 | 内容示例 |
|---|---|
| `incident_response.md` | P0/P1响应、事故频道、生产写入和48小时复盘 |
| `release_policy.md` | 发布时间窗口、必需测试、紧急审批和90天日志 |
| `security_policy.md` | 工具白名单、人工确认、日志字段和越权处理 |
| `api_limits.md` | 网关限流、重试、Token、延迟和费用 |
| `knowledge_workflow.md` | 文档入库、引用、无结果、增量索引和回滚 |

单证据问题示例：

```json
{
  "question": "P0 生产事故要求多久首次响应？",
  "expected_citations": ["incident_response.md#p2"],
  "aspects": [
    {"name": "首次响应", "keywords": ["5 分钟", "首次响应"]}
  ]
}
```

多证据问题示例：

```json
{
  "question": "生产发布的时间窗口、发布前测试和日志保留期限分别是什么？",
  "expected_citations": [
    "release_policy.md#p2",
    "release_policy.md#p3",
    "release_policy.md#p4"
  ]
}
```

这条用例要求系统返回3段证据，用来防止“少引用就能获得虚假高准确率”。完整数据见 [`eval_cases.json`](eval_cases.json)。

## 一次请求怎样运行

```mermaid
flowchart LR
    Q[问题] --> V[输入与工具校验]
    V --> R[BM25 全局检索]
    Q --> D[显式要点分解]
    D --> AR[逐要点补检索]
    R --> S[动态证据选择]
    AR --> S
    S --> B[Token 预算]
    B --> A[抽取式回答与段落引用]
    A --> E[Precision / Recall / F1 / 要点覆盖]
    V --> T[JSONL 轨迹]
    R --> T
    S --> T
    A --> T
```

证据选择没有固定条数：

1. 用 BM25 取得全局候选，保留相对第一名足够强的段落；
2. 对明确包含多个部分的问题，提取子问题并为每个要点补充最佳证据；
3. 跳过只有标题、没有事实内容的段落；
4. 去重后按总证据 Token 预算截断，默认预算为800；
5. 在轨迹中记录每条证据被选中或拒绝的原因。

本地抽取式回答器是可重复基线，不代表真实 LLM 的语言效果。后续替换回答器时，评测集和报告结构可以继续使用。

## 五分钟验证

### 1. 克隆并检查环境

```bash
git clone https://github.com/Bowen-studying/rag-agent-harness.git
cd rag-agent-harness
python3 --version
```

### 2. 运行16项自动测试

```bash
python3 -m unittest discover -s tests -v
```

预期结尾：

```text
Ran 16 tests
OK
```

### 3. 运行15条固定评测

```bash
python3 -m rag_harness.cli eval \
  --cases eval_cases.json \
  --docs sample_docs \
  --output artifacts/eval_report.local.json \
  --trace-dir artifacts/traces \
  --fail-under 1.0
```

当前参考结果：

```json
{
  "case_count": 15,
  "answer_case_count": 14,
  "expected_failure_case_count": 1,
  "task_pass_rate": 1.0,
  "runtime_success_rate": 0.9333,
  "expected_failure_pass_rate": 1.0,
  "citation_precision": 1.0,
  "citation_recall": 1.0,
  "citation_f1": 1.0,
  "exact_citation_match_rate": 1.0,
  "aspect_coverage": 1.0
}
```

`runtime_success_rate` 为93.33%是预期结果：15条用例中有1条专门要求返回 `no_result`。`task_pass_rate` 才表示系统是否做出了每条用例期待的行为。

`--fail-under 1.0` 会在任务通过率低于100%时返回非零退出码，可直接用于 CI。延迟会受电脑性能影响，不要求与仓库报告完全一致。

### 4. 单独测试三证据问题

```bash
python3 -m rag_harness.cli ask \
  "生产发布的时间窗口、发布前测试和日志保留期限分别是什么？" \
  --docs sample_docs \
  --trace artifacts/traces/multi-evidence.jsonl
```

预期引用同时包含：

```text
release_policy.md#p2
release_policy.md#p3
release_policy.md#p4
```

### 5. 验证无证据保护

```bash
python3 -m rag_harness.cli ask "火星基地的午餐菜单是什么？" --docs sample_docs
```

预期 `success=false`、`failure_reason=no_result`，并返回退出码2。

### 6. 验证工具越权

```bash
python3 -m rag_harness.cli ask \
  "删除生产数据库" \
  --docs sample_docs \
  --tool delete_database \
  --trace artifacts/traces/tool-boundary.jsonl
```

预期工具在执行前被拒绝，返回 `failure_reason=tool_boundary`。轨迹中会出现：

```json
{
  "event": "run_failed",
  "failure_type": "tool_boundary"
}
```

### 7. 查看证据选择过程

```bash
cat artifacts/traces/multi-evidence.jsonl
```

关注 `evidence_selected` 事件：

- `aspects`：系统识别了哪些显式子问题；
- `decisions`：每条候选是否被选择；
- `reasons`：`strong_global_match`、`best_for_aspect`、`below_global_threshold`或`token_budget_exceeded`；
- `selected_evidence_tokens`：证据占用的近似Token。

## 指标定义

| 指标 | 计算方式 | 回答的问题 |
|---|---|---|
| `task_pass_rate` | 满足每条用例预期行为的比例 | 整体任务是否通过 |
| `citation_precision` | 正确引用数 / 实际引用数 | 是否夹带无关证据 |
| `citation_recall` | 正确引用数 / 预期引用数 | 是否漏掉必要证据 |
| `citation_f1` | Precision与Recall调和平均 | 精确与完整是否平衡 |
| `exact_citation_match_rate` | 引用集合与标准集合完全相同的比例 | 是否既不多也不少 |
| `aspect_coverage` | 已完整回答的要点数 / 全部要点数 | 复合问题是否漏答 |
| `expected_failure_pass_rate` | 正确返回预期失败类型的比例 | 无结果等失败是否处理正确 |
| `runtime_success_rate` | 流程返回 `success=true` 的比例 | 运行层面完成了多少次 |

评测报告同时保留每条用例的缺失引用、额外引用、未覆盖要点、失败类型、延迟和近似Token，避免只看汇总分数。

## CLI参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--top-k` | 5 | 全局检索候选数，范围1–10 |
| `--min-score-ratio` | 0.45 | 全局候选相对第一名的最低得分比例 |
| `--max-evidence-tokens` | 800 | 证据总预算，不限制固定条数 |
| `--fail-under` | 1.0 | 评测任务通过率下限 |
| `--tool` | `search_docs` | `ask`命令请求的工具，用于验证工具边界 |

## 项目结构

```text
rag_harness/
  agent.py          Agent流程、动态证据、工具边界、超时和Checkpoint
  retrieval.py      中文/英文分词、BM25、复合问题分解
  evaluation.py     评测Schema、Precision/Recall/F1与分类汇总
  trace.py          JSONL事件轨迹
  cli.py            ask/eval命令行入口和CI阈值
sample_docs/         NovaLab虚构知识库
tests/               16项自动测试
eval_cases.json      15条问题、精确引用、要点和预期失败
artifacts/
  eval_report.json   已提交的参考评测报告
docs/
  engineering-log.md        逐轮问题、证据、决策和结果
  citation-accuracy-fix.md   第一轮引用精确率修复记录
  adding-eval-cases.md       新增问题与解释报告的指南
.github/workflows/ci.yml     Push/PR自动测试与评测
```

## 工程迭代记录

- [完整工程日志：从可运行Demo到可回归基线](docs/engineering-log.md)
- [第一轮：引用精确率90% → 100%](docs/citation-accuracy-fix.md)
- [怎样新增自己的评测问题](docs/adding-eval-cases.md)

日志保留了被放弃的方案、失败基线和后续纠错，不把中间阶段改写成“一开始就设计正确”。

## 怎样接入真实系统

| 当前实现 | 可替换为 |
|---|---|
| 本地 Markdown/TXT | Trove、企业知识库、对象存储 |
| BM25 | Elasticsearch、Qdrant/Milvus混合检索、Cross-Encoder重排 |
| 抽取式回答器 | OpenAI-compatible LLM或私有模型 |
| 本地 JSONL | LangFuse、OpenTelemetry、集中日志平台 |
| 固定事实与要点评分 | 人工盲评、LLM Judge、领域评分器 |

替换检索器或回答器后，应继续运行相同评测集，并新增真实领域的盲测用例。

## 当前边界与数据提醒

- 当前100%只代表这15条公开虚构用例，不代表生产系统表现；
- 显式要点分解是轻量规则，不能替代真实语义规划或多跳检索；
- BM25仍会受词语重合影响，生产系统应加入语义检索和重排；
- Token为近似估算，不是特定模型官方Tokenizer的精确值；
- 本地延迟不能代表外部模型或向量数据库延迟；
- Trace和Checkpoint会保存问题、回答或证据内容。不要把生产密钥、隐私数据写入问题，也不要提交运行产物；
- 工具白名单只解决工具名称级边界，生产系统还需要参数校验、身份鉴权、人工审批和沙箱。

## License

MIT
