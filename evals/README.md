# 评测集说明

`baselines.json` 是评测基线清单。当前默认基线是 `pdf_gold_current`，对应 `current_pdf_eval_set.json`。调用 `/api/evaluation/run` 或 CLI 未显式指定套件时，默认只跑这个当前可信的 PDF gold 基线。

`current_pdf_eval_set.json` 是当前 active gold 基线，包含 64 条 PDF-only case。它带有本地 `expected_chunk_ids`，用于计算真正的 gold chunk `recall@1/3/5/k`，并在每条结果中记录命中的 `gold_chunk_hit_ids`、漏掉的 `gold_chunk_missed_ids`。这些 chunk id 绑定当前 Chroma 索引里的 document_id，适合证明当前环境的检索表现，不作为跨机器通用模板。已经声明 `expected_pages` 的 case，页码必须和当前索引中 gold chunk 的 `page_start/page_end` 范围一致；没有声明 `expected_pages` 的多文档或拒答 case 不做页码门禁。

`current_network_programming_eval_set.json` 是基于两份“网络程序设计”DOCX 报告制作的领域基线，目前在 `baselines.json` 中标记为 `paused`。当前本地 ready 文档缺少这两份 DOCX，因此它不能进入默认均分。

`current_mixed_pdf_docx_eval_set.json` 是 PDF/DOCX 混合集成基线，也暂时标记为 `paused`。恢复 DOCX 文档并重新索引后，再用它验证跨 PDF/DOCX、多文档关系、视觉和拒答能力。

`sample_eval_set.json` 只作为 smoke 测试，用于检查评测通路是否能跑，不作为真实质量分数。

`templates/mixed_pdf_docx_regression_template.json` 是 PDF/DOCX 混合评测模板，不作为真实成绩基线。

## 评测目标

这套评测不是只看“回答顺不顺”，而是同时检查：

- 检索是否命中正确上下文：`retrieval_hit`、`context_precision`、`context_recall`
- 标准证据 chunk 是否被召回：`gold_chunk_recall_at_1`、`gold_chunk_recall_at_3`、`gold_chunk_recall_at_5`、`gold_chunk_recall_at_k`、`gold_chunk_hit_ids`、`gold_chunk_missed_ids`
- 标准证据是没召回，还是召回后被过滤：`gold_chunk_candidate_recall_at_k`、`trace_summary.gold_evidence.dropped_after_retrieval_ids`、`trace_summary.gold_evidence.not_retrieved_ids`
- 引用是否可靠：`citation_hit`、`citation_accuracy`
- 引用页码为什么没命中：`trace_summary.citation_pages`，其中包含 `expected_pages`、`matched_pages`、`missed_pages`、可见证据页码区间和最近证据页距离
- 多文献是否覆盖清楚：`document_coverage`
- 图片/图表证据是否命中：`image_evidence_hit`、`visual_evidence_hit`、`visual_summary_hit`
- OCR 是否真的命中文本：`ocr_evidence_hit`、`ocr_text_hit`
- Embedding 是否发生备用检索降级：`embedding_fallback_count`、`embedding_fallback_rate`
- 评测分数是否可作为可信基线：`evaluation_trustworthy`、`trust_gate_status`、`trust_gate_failures`
- 回答是否切题并忠于证据：`answer_relevance`、`faithfulness_proxy`；其中 `faithfulness_proxy` 是本地句子-证据支撑度代理，不等同于 LLM judge 的语义忠实度评分
- LLM-as-judge 语义评分：`judge_score`、`judge_scores`
- 工程表现：`latency_ms`

## LLM-as-judge

如果 `.env` 中 `ENABLE_LLM_JUDGE=true`，评测会额外调用 `JUDGE_MODEL` 作为评委模型。评委模型只看本轮问题、答案、证据和 trace，不允许凭常识补答案。

Judge 输出以下分数，范围都是 0-1：

- `answer_relevance`：是否答到问题
- `faithfulness`：是否忠于证据
- `citation_support`：引用是否真的支撑句子
- `context_usage`：是否使用了相关证据，避免混入无关上下文
- `multi_document_clarity`：多文献关系是否分清楚
- `visual_grounding`：图片/图表证据是否用对
- `completeness`：是否覆盖用户要求
- `no_hallucination`：是否没有编造

最终 `score` 会在有 judge 时按 55% 程序化指标 + 45% judge 分数组合；没有 judge 时只使用程序化指标。无论是否启用 judge，程序化硬上限都会最终生效，避免 gold evidence 漏召回却被 judge 高分抬成高可信结果。

## Case 字段

- `id`：评测用例 ID
- `question`：用户问题
- `expected_keywords`：答案中希望覆盖的关键词
- `expected_answer`：可选，给 judge 的参考答案，不参与硬匹配
- `expected_evidence_keywords`：证据中希望覆盖的关键词
- `expected_document` / `expected_documents`：希望命中的文档名片段
- `expected_page` / `expected_pages`：希望命中的页码；在 active PDF gold 基线中，它必须能被 `expected_chunk_ids` 对应 chunk 的 `page_start/page_end` 范围解释
- `required_document_count`：多文献问题至少应覆盖的文档数量
- `expected_modalities`：可填 `image`、`figure`、`chart`、`table`、`vision`
- `relation_keywords`：多文献关系类问题希望出现的关系词
- `judge_rubric`：给 judge 的额外评分要求

## 建议用例规模

真正判断项目效果，建议至少准备 30-80 条 case：

- 单文档事实问答：8-12 条
- 方法/结论/局限概括：8-12 条
- 多文献共同点/差异/互补/冲突：8-15 条
- PDF 图片、DOCX 图片、表格、截图证据：6-12 条
- 无证据或证据不足时拒答：4-8 条
- 格式稳定性，如表格、引用、分段：4-8 条

每次改 prompt、检索策略、图像处理、chunk 参数后，都跑一次同一套评测，对比 `avg_score`、`avg_judge_score`、`avg_context_precision`、`avg_document_coverage` 和 `avg_citation_accuracy`。

当前 PDF gold 基线的侧重点：

- PDF 正文事实、方法、结论和边界
- PDF 表格、图像/视觉、扫描型 PDF 和 OCR
- 多 PDF 关系、术语边界和拒答
- gold chunk 命中、漏召回、候选召回与最终证据过滤

DOCX 相关基线目前暂停，原因是当前本地 ready 文档缺少网络程序设计实验一/实验二。恢复这两份文档后，需要先跑 `python -m backend.app.eval_audit --local`，确认 `local_missing_document` 清零，再恢复它们的基线地位。

## 调用方式

查看分层基线：

```bash
python -m backend.app.eval_cli --list-baselines
```

默认运行当前 active gold 基线，并且只使用该基线 `expected_document` 对应的 ready 文档：

```bash
python -m backend.app.eval_cli
```

API 默认同样使用 `baselines.json` 的 `default_baseline`：

```json
{
  "enable_judge": true
}
```

也可以显式指定基线或只跑部分用例：

```json
{
  "baseline_id": "pdf_gold_current",
  "case_ids": ["pdf_nist_ai_rmf_core_functions"],
  "enable_judge": true
}
```

如果明确要调试暂停基线，可以显式传入 `baseline_id`，但这类结果不能和 active gold 基线混入同一个均分。

结果会写到 `data/eval_runs/`，同时在 `data/observability/eval_runs.jsonl` 和 `data/observability/eval_case_runs.jsonl` 留下可追踪记录，方便做横向对比。

## 评测集质量审计

正式跑分前，先运行评测集审计：

```bash
python -m backend.app.eval_audit --local
```

审计会检查 case ID/问题是否重复、字段是否被系统忽略、答案关键词是否过泛、证据锚点是否不足、视觉/表格/OCR 问题是否声明了对应模态、本地 ready 文档是否覆盖 `expected_document`，以及 `expected_chunk_ids` 是否真的存在于本地 Chroma 索引。

其中 `error` 级别问题会直接影响评测可信度，应先处理；`warning` 级别通常表示评分锚点不够硬，适合在扩充 gold evidence 或细化 expected claims 时逐步收敛。

## 分级报告

每条评测结果会给出 `result_status`：

- `pass`：关键指标通过，没有明显失败项
- `warn`：可运行但存在弱项，例如分数低于通过线、gold chunk 部分覆盖或引用略弱
- `fail`：回答完成了，但质量不可信，例如召回失败、证据过滤失败、引用失败、拒答失败、视觉/OCR 失败
- `blocked`：case 没有进入可信评分条件，例如模型调用失败、运行异常或发生 embedding fallback

每条结果还会包含 `failure_categories`、`grading_reasons` 和 `grading_report`。整轮结果会汇总 `result_status_counts`、`failure_category_counts` 和 `grading_summary`，用于快速判断本轮主要问题集中在哪里。若任一 case 发生 embedding fallback，整轮 `evaluation_trustworthy=false` 且 `trust_gate_status=not_comparable`，本轮均分只能用于诊断，不能作为可信基线或横向对比成绩。

当前主要失败类别：

- `retrieval_failure`：期望文档或 gold chunk 没有进入候选召回
- `evidence_filtering_failure`：gold chunk 进入候选，但没进入最终回答证据
- `citation_failure`：引用无效、引用页码不对或引用准确率过低
- `refusal_failure`：应拒答没拒答，或不该拒答时错误拒答
- `visual_ocr_failure`：视觉/OCR case 没有命中可用图像摘要或 OCR 文本
- `table_evidence_failure`：表格 case 没有命中表格证据
- `faithfulness_failure` / `claim_failure`：回答主张覆盖不足或证据支撑不足
- `embedding_fallback`：评测使用了备用检索向量空间，本轮分数不可作为可信基线

## 建议通过线

- `evaluation_trustworthy = true` 且 `trust_gate_status = passed`：本轮分数可以进入可信基线对比。
- `avg_score >= 0.75`：当前文档集可认为基本可用。
- `grading_summary.pass_rate >= 0.75` 且 `result_status_counts.blocked = 0`：本轮评测整体可解释。
- `avg_context_precision >= 0.65`：检索证据整体相关。
- `avg_gold_chunk_recall_at_k >= 0.75`：标准证据 chunk 能稳定进入最终回答证据。
- `avg_gold_chunk_candidate_recall_at_k >= 0.85`：标准证据 chunk 至少能稳定出现在候选召回轨迹中；若该指标高但最终 recall 低，优先检查证据过滤/裁判环节。
- `avg_document_coverage >= 0.85`：多文档问题没有明显漏文档。
- `avg_citation_accuracy >= 0.85`：回答引用基本有效。
- `avg_judge_score >= 0.75` 且 `judge_coverage = 1`：语义质量通过 LLM-as-judge 基线。

如果图片相关 case 的 `image_evidence_hit` 或 `judge_visual_grounding` 偏低，优先检查视觉模型配置、图片 chunk 是否入库，以及证据展示是否只展示相关图片。
