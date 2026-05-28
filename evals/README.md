# 评测说明

默认评测现在只做“轻量健康检查 + 证据准确性守门”，不再追求复杂跑分。

默认基线是 `sample_smoke`，对应 `sample_eval_set.json`。它检查：

- 问答是否能完成。
- 是否返回证据。
- 回答中是否引用了可点击证据编号。
- 如用例指定文档数量或文档名，证据是否覆盖到位。
- 如用例指定证据关键词，被引用证据是否命中关键依据。
- 是否出现 embedding fallback。

默认运行：

```bash
python -m backend.app.eval_cli
```

输出只保留通过率、失败原因、证据数、引用数、embedding fallback 和结果文件路径。复杂 PDF/DOCX 评测集和旧评分代码已删除，不再保留未接入的评测功能。

## 长期 RAG Gold 基准

`rag_gold_v1` 是长期回归评测，不替代默认 smoke。它使用 3 篇论文、30 条 case，检查 gold evidence、答案要点、引用支撑、拒答、多文档和图表相关问题。

准备评测文档：

```bash
mkdir -p data/eval_documents
curl -L -o data/eval_documents/attention-is-all-you-need.pdf https://arxiv.org/pdf/1706.03762
curl -L -o data/eval_documents/bert.pdf https://arxiv.org/pdf/1810.04805
curl -L -o data/eval_documents/unet.pdf https://arxiv.org/pdf/1505.04597
```

文档清单和 sha256 见 `rag_gold_documents.json`。确保 3 篇文档都已通过项目索引，并且 embedding 模型一致后运行：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost python -m backend.app.eval_cli --baseline rag_gold_v1
```

核心指标：

- `avg_retrieval_recall_at_k`：检索证据命中 gold evidence 的比例。
- `avg_citation_support_rate`：回答引用是否支撑 gold evidence。
- `avg_answer_point_coverage`：回答覆盖 expected answer points 的比例。
- `embedding_fallback_rate`：是否退回本地 embedding，非 0 时结果不适合做横向比较。

## Unseen Paper Generalization Baseline

`generalization_gold_v1` is a supporting gold benchmark for checking generalization on papers that are not part of the original demo set. It uses 4 additional papers and 10 cases covering method explanation, benchmark results, training/data factors, efficiency claims, multi-document comparison, and unsupported cross-domain claims.

Prepare documents:

```bash
mkdir -p data/eval_documents
curl -L -o data/eval_documents/efficientnet.pdf https://arxiv.org/pdf/1905.11946
curl -L -o data/eval_documents/focal-loss.pdf https://arxiv.org/pdf/1708.02002
curl -L -o data/eval_documents/simclr.pdf https://arxiv.org/pdf/2002.05709
curl -L -o data/eval_documents/lora.pdf https://arxiv.org/pdf/2106.09685
```

Index those PDFs in the app, then run:

```bash
python -m backend.app.eval_cli --baseline generalization_gold_v1 --write-report
```

The CLI writes the normal JSON result and, with `--write-report`, a Markdown summary in `data/eval_runs`.

To diagnose where gold evidence is lost without changing retrieval behavior, add `--audit-gold`:

```bash
python -m backend.app.eval_cli --baseline generalization_gold_v1 --write-report --audit-gold
```

The audit table reports how many gold evidence items appeared in fused candidates, retrieval-selected evidence, answer prompt evidence, visible evidence, and cited evidence.
