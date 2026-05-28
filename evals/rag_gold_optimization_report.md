# RAG Gold 评测优化记录

本文记录 `rag_gold_v1` 长期评测基准的两次主要评分结果，以及为提升评分执行的结构性改动。

## 评测环境

- 评测集：`rag_gold_v1`
- Case 数量：30
- 文档：
  - `attention-is-all-you-need.pdf`
  - `bert.pdf`
  - `unet.pdf`
- 对话模型：项目当前默认 chat model
- Embedding 模型：`doubao-embedding-vision-250615`
- 运行命令：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost .venv/bin/python -m backend.app.eval_cli --baseline rag_gold_v1
```

## 两次评分对比

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| run_id | `eval_feb1d3ca586e40e9` | `eval_d0fa591094714d56` | - |
| case_count | 30 | 30 | 0 |
| pass_count | 2 | 4 | +2 |
| fail_count | 28 | 26 | -2 |
| pass_rate | 6.7% | 13.3% | +6.6pp |
| avg_retrieval_recall_at_k | 42.8% | 53.9% | +11.1pp |
| avg_citation_support_rate | 24.3% | 25.4% | +1.1pp |
| avg_answer_point_coverage | 54.4% | 57.8% | +3.4pp |
| avg_document_coverage | 88.9% | 93.3% | +4.4pp |
| avg_citation_accuracy | 96.7% | 96.7% | 0 |
| embedding_fallback_count | 0 | 0 | 0 |
| embedding_fallback_rate | 0.0% | 0.0% | 0 |
| evaluation_trustworthy | true | true | - |
| avg_latency_ms | 9070 | 8433 | -637ms |

## 失败类型对比

| 失败类型 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| `gold_evidence_missed` | 17 | 14 | -3 |
| `citation_unsupported` | 18 | 24 | +6 |
| `answer_point_missing` | 22 | 21 | -1 |
| `wrong_document` | 6 | 3 | -3 |
| `no_evidence` | 1 | 1 | 0 |
| `citation_missing` | 1 | 1 | 0 |
| `refusal_missing` | 1 | 0 | -1 |

## 本轮执行的结构性改动

1. 构建长期 Gold 评测基准。
   - 新增 `evals/rag_gold_eval_set.json`，覆盖 3 篇论文、30 条 case。
   - 新增 `gold_evidence`、`expected_answer_points`、`case_type`、`difficulty` 等评测字段。
   - 新增 `rag_gold_v1` baseline。

2. 扩展评测指标。
   - 增加 `avg_retrieval_recall_at_k`。
   - 增加 `avg_citation_support_rate`。
   - 增加 `avg_answer_point_coverage`。
   - 保留 `embedding_fallback_rate` 作为可信度守门指标。

3. 优化检索召回。
   - 增加 retrieval query decomposition。
   - 原问题作为主查询走 dense + BM25 + 规则检索。
   - 扩展查询只走 BM25/规则检索，避免多次远程 embedding 调用导致 fallback。
   - 多文档问题不再因文档名命中而误收窄检索范围。

4. 优化证据选择。
   - 增加 role-balanced evidence selection。
   - 不再只按综合分 top-k 取证据，而是按问题类型保底选择证据角色。
   - 对 summary、method、result、figure、training、refusal 等问题使用不同证据角色词。

5. 优化回答与引用约束。
   - 增强 answer contract，要求结论加证据要点。
   - 要求关键事实绑定 `[E#]`。
   - 对多文档、结果指标、图表、拒答问题加入更明确的输出约束。
   - 增加基于已有证据的回答后补齐逻辑。

6. 优化评测运行稳定性。
   - 查询侧 embedding 若 fallback 到本地向量，且目标索引不是本地索引，则跳过 dense 查询，避免 384 维查询向量混入 2048 维 Chroma 集合。
   - 最终有效评测保持 `embedding_fallback_count = 0`。

## 结果判断

本轮优化对检索召回和文档覆盖有明显帮助：

- `avg_retrieval_recall_at_k` 从 42.8% 提升到 53.9%。
- `avg_document_coverage` 从 88.9% 提升到 93.3%。
- `wrong_document` 从 6 降到 3。

但没有达到原计划的 30% 通过率目标。主要瓶颈仍然是：

- `citation_unsupported` 仍然很高，说明 chunk 级引用过粗，回答引用的证据未必包含支撑句。
- `answer_point_missing` 仍然很高，说明生成阶段还没有稳定覆盖 expected answer points。
- 图表、结果、跨文档问题仍然弱。

下一步建议不要继续堆 prompt 或规则，而是引入句级/表格行级 evidence claim：

```text
chunk -> evidence sentences / table rows -> rerank -> answer only from selected claims
```

这样才能显著提高引用支撑率和答案覆盖率。
