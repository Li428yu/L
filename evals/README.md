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
