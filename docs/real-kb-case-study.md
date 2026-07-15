# 从10题初测到Schema 3.0：真实知识库案例

## 背景与系统边界

本案例把公开的NovaLab Harness接到本地PDF知识库和Obsidian Vault。资料链路不是“Trove和Obsidian并列检索”：Trove负责存储、OCR、AI摘要、语义搜索与AI问答，Trove Sync把处理后的内容同步到Obsidian；Obsidian用于离线阅读、标注、链接笔记和知识网络。Harness读取PDF镜像与Obsidian Markdown，补充稳定引用、固定评测、失败分类、轨迹和同步门槛。

本地索引包含69份PDF和104份Obsidian Markdown，共173份文档。真实原文、Vault、文件名、绝对路径、私有用例和原始轨迹都不进入GitHub。

## 部署初测：数字可运行，但口径不可信

旧版用10条问题得到：

| 指标 | 初测值 | 后续发现 |
|---|---:|---|
| 引用Precision | 100% | 把检索候选当成最终引用，且期望引用受检索结果影响 |
| 引用Recall | 90.7% | `#pN`实际为TXT段落序号，不是PDF物理页码 |
| 引用F1 | 94.1% | 继承了上述两个口径问题 |
| 任务通过率 | 60% | 混合了检索、抽取式回答和关键词匹配失败 |
| 平均延迟 | 约10ms | 只统计BM25，没有包含索引加载和模型判断 |

这组结果只保留为“部署初测”，不能作为当前效果，也不能和NovaLab受控100%横向比较。

## 问题定位与修正

### 1. 段落序号不是页码

旧TXT按空行分段后生成`#pN`。例如矛盾同一性与斗争性的真实证据位于PDF物理第52页，旧引用却显示为100以上的段落号。Schema 3.0直接逐页读取PDF，长页只在页内继续分块，引用结构改为：

```text
doc_id + pdf-page=N + chunk=M
```

Obsidian使用：

```text
doc_id + heading=标题路径 + chunk=M
```

### 2. 黄金证据不能从系统答案反推

旧流程是“先搜索，再把返回页写进expected citations”，会让系统给自己出答案。新版先由人阅读PDF页或笔记小节，再写`gold_evidence_groups`；每个要点可接受一个或多个等价位置，并绑定`source_manifest_sha256`。来源内容改变后，旧标注不能静默沿用。

### 3. BM25候选不是最终引用

BM25先取Top 20，再按显式要点、标题噪声、相对分数和Token预算形成最多8个候选。候选层的目标是少漏证据，不应直接用Citation Precision评价。外部Agent对脱敏候选做相关性判断，Harness再验证选中的ID一定来自候选集合。

因此报告同时给出：

- `candidate_group_recall`：黄金证据有没有进入候选；
- `evidence_precision / evidence_group_recall / evidence_f1`：语义层最终选证据的质量；
- `hard_negative_accuracy`：共享词汇但无法回答的问题是否拒答；
- `semantic_decision_coverage`：是否真的运行了语义层，而不是用词法状态冒充。

### 4. 随机字符串不是真实边界测试

旧负例`zzzabc123`只能证明“完全没有词命中”。新版6条自然硬负例保留领域词，例如CAN SLIM、RAG、ComfyUI和Gibbs phase rule，同时询问知识库未覆盖的医学、法律或组织决策内容。它们会进入BM25候选，必须由语义层识别为证据不足或需要澄清。

### 5. 同步必须有质量门槛

旧监控按文件名和修改时间判断变化，重新生成后即覆盖，还使用`--fail-under 0.0`。新版按实际内容SHA256增量重建，删除源文件时清理孤立索引，并执行：

```text
候选索引 → 固定评测 → 质量门槛 → 原子替换
                         ↘ 失败时保留旧索引
```

## 当前真实评测设计

- 32条检索用例：26条可回答、6条自然硬负例；
- 覆盖5本根目录书籍、AI产品经理、AI漫剧和Obsidian笔记；
- 多条问题包含2—7个显式证据要点；
- 8条费曼反馈用例：正确2、遗漏2、事实错误2、知识库不足/问题含糊2；
- 费曼反馈只评错误类型、遗漏识别、引用有效性和输出Schema，不要求自由文本逐字一致。

## 修正后实测结果

| 指标 | Schema 3.0结果 | 说明 |
|---|---:|---|
| 执行成功率 | 100% | 32题均完成检索和判断流程 |
| 候选Group Recall | 100% | 人工黄金证据全部进入BM25候选 |
| 最终证据Precision / Recall / F1 | 92.31% / 92.31% / 92.31% | 只计算Flash实际选择的证据，不把全部候选当引用 |
| 要点证据覆盖率 | 92.31% | 复合问题按证据组统计 |
| 自然硬负例正确拒答率 | 100% | 6/6共享词汇负例均拒答 |
| 检索任务通过率 | 93.75% | 30/32题满足预期行为 |
| 引用定位有效率 | 100% | 所有选择ID都能回到当前索引 |
| 语义判断覆盖率 | 100% | 32题均实际运行`deepseek-v4-flash` |

两条失败都不是BM25漏检：黄金证据已进入候选，但Flash把可回答问题误判为`insufficient_evidence`。因此下一轮应比较语义判断Prompt和模型，而不是用增加Top K掩盖问题。索引加载约2.31秒，BM25 p50约109.6毫秒；二者分开报告，不再用旧版“约10ms”代表整条链路。

完整结果见[`../artifacts/kb_eval_report.public.json`](../artifacts/kb_eval_report.public.json)。该报告不含原文、文件名、绝对路径、用户标识或原始轨迹。

## 匿名费曼示例：先有同一性，后有斗争性？

学生口述：“同一性就是相互依存，斗争性就是相互对抗。先有同一性后有斗争性。”

系统应分开处理：

- “相互依存”“相互对抗”属于已有准确部分；
- “相互贯通”“相互分离”属于遗漏；
- “先有同一性后有斗争性”属于事实冲突，因为证据说明二者相互联结、相辅相成；
- 反馈引用PDF物理第52页，并要求学生重新解释二者关系。

这个案例不评价文风是否漂亮，只检查偏差分类与引用是否能回到真实证据。

## 可复现入口

- [`kb_sources.example.toml`](../kb_sources.example.toml)：无个人路径的来源配置；
- [`eval_kb_cases.public.json`](../eval_kb_cases.public.json)：匿名32题结构；
- [`eval_feynman_cases.public.json`](../eval_feynman_cases.public.json)：8条费曼反馈用例；
- [`artifacts/kb_eval_report.public.json`](../artifacts/kb_eval_report.public.json)：真实脱敏报告；
- [`tests/test_v3.py`](../tests/test_v3.py)：连接器、稳定引用、语义ID边界、同步和隐私测试。
