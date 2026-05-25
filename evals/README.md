# 评测集说明

`sample_eval_set.json` 是默认回归评测集。调用 `/api/evaluation/run` 时，系统会用当前 RAG pipeline 逐条提问，并输出本地 JSON 结果；如果开启 Langfuse，也会同步写入 trace 和 score。

`current_network_programming_eval_set.json` 是基于当前已上传且已索引的两份“网络程序设计”DOCX 报告制作的真实基线集，目前包含 43 条 case，比默认模板集更适合判断现在这批文档的实际完成效果。

`templates/mixed_pdf_docx_regression_template.json` 是 PDF/DOCX 混合评测模板，目前不作为真实成绩基线。等 PDF 文档完成上传、解析、图片视觉摘要入库和向量索引后，再把模板中的占位文档名、页码和关键词替换成真实内容。

`current_mixed_pdf_docx_eval_set.json` 是基于 8 份公开 PDF 与当前两份 DOCX 构建的真实混合评测集，覆盖 PDF 正文、表格、图片/视觉、扫描型 PDF、多文档关系和证据不足拒答。运行时建议显式传入评测语料对应的 document_ids，避免把私人上传文档混入评测上下文。

## 评测目标

这套评测不是只看“回答顺不顺”，而是同时检查：

- 检索是否命中正确上下文：`retrieval_hit`、`context_precision`、`context_recall`
- 引用是否可靠：`citation_hit`、`citation_accuracy`
- 多文献是否覆盖清楚：`document_coverage`
- 图片/图表证据是否命中：`image_evidence_hit`
- 回答是否切题并忠于证据：`answer_relevance`、`faithfulness_proxy`
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

最终 `score` 会在有 judge 时按 55% 程序化指标 + 45% judge 分数组合；没有 judge 时只使用程序化指标。

## Case 字段

- `id`：评测用例 ID
- `question`：用户问题
- `expected_keywords`：答案中希望覆盖的关键词
- `expected_answer`：可选，给 judge 的参考答案，不参与硬匹配
- `expected_evidence_keywords`：证据中希望覆盖的关键词
- `expected_document` / `expected_documents`：希望命中的文档名片段
- `expected_page` / `expected_pages`：希望命中的页码
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

当前真实集的侧重点：

- 单文档 DOCX 问答、实验流程、代码/API、调试与改进
- DOCX 内嵌图片/视觉摘要证据，检查“只引用相关图片”
- 表格证据，检查评分表分项和总分
- 多文档共同点、差异、互补、冲突和同一术语不同语义
- 无证据拒答，检查 UDP、并发、性能基准、安全加密、异步套接字等未被文档证明的主张

当前真实集还不能直接代表 PDF 效果，因为可检索文档表里只有两份 ready DOCX。PDF 评测需要先让 PDF 成为 ready 文档，再把模板升级为真实 case。

## 调用方式

默认会使用所有已就绪文档：

```json
{
  "suite_name": "current_network_programming_eval_set",
  "enable_judge": true
}
```

也可以只跑部分用例：

```json
{
  "suite_name": "current_network_programming_eval_set",
  "case_ids": ["multi_doc_compare", "tcp_visual_run_result"],
  "enable_judge": true
}
```

结果会写到 `data/eval_runs/`，同时在 `data/observability/eval_runs.jsonl` 和 `data/observability/eval_case_runs.jsonl` 留下可追踪记录。开启 Langfuse 时，每条 case 也会单独写入 trace 和 score，方便做横向对比。

## 建议通过线

- `avg_score >= 0.75`：当前文档集可认为基本可用。
- `avg_context_precision >= 0.65`：检索证据整体相关。
- `avg_document_coverage >= 0.85`：多文档问题没有明显漏文档。
- `avg_citation_accuracy >= 0.85`：回答引用基本有效。
- `avg_judge_score >= 0.75` 且 `judge_coverage = 1`：语义质量通过 LLM-as-judge 基线。

如果图片相关 case 的 `image_evidence_hit` 或 `judge_visual_grounding` 偏低，优先检查视觉模型配置、图片 chunk 是否入库，以及证据展示是否只展示相关图片。
