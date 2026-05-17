# 项目结构说明

这个项目现在按“页面层”和“Agent 能力层”拆开，读代码时可以按下面顺序走。

## 入口

- `app.py`：Streamlit 页面入口，只负责组装页面、创建 `PaperAssistant`、处理一次提问的流式执行。
- `paper_assistant.py`：兼容导出层，外部仍然可以从这里导入 `PaperAssistant` 和相关数据类型。

## 页面层：`ui/`

- `ui/state.py`：Streamlit session state、当前轮记录、历史轮次、runtime 焦点轮次。
- `ui/setup_view.py`：上传论文、建立索引、索引概览、生成摘要卡片。
- `ui/indexing.py`：从上传文件构建 chunk、embedding 向量和论文概览。
- `ui/conversation_view.py`：左侧历史对话、中间当前对话、快捷提问和运行中预览。
- `ui/runtime_view.py`：右侧 LangGraph 流程图、节点日志、工具调用、检索证据和 Mermaid 源码。

## Agent 能力层：`assistant_core/`

- `assistant_core/assistant.py`：`PaperAssistant` 主类，只负责初始化模型客户端和组合 mixin。
- `assistant_core/types.py`：`Chunk`、`RetrievedChunk`、`ToolTrace`、`AgentTurnResult` 等数据结构。
- `assistant_core/documents.py`：PDF/DOCX 文本提取、chunk 切分、论文摘要卡生成。
- `assistant_core/retrieval.py`：embedding、向量检索、多模态 embedding 兼容和重试。
- `assistant_core/planner.py`：planner 节点的规则，决定先调用哪个工具。
- `assistant_core/graph.py`：LangGraph 节点、边、`ToolNode`、流式 runtime 事件。
- `assistant_core/tracing.py`：把 LangGraph 消息转成页面可展示的节点日志、工具调用和证据。
- `assistant_core/utils.py`：成本估算等小工具函数。

## 阅读建议

1. 先看 `app.py`，理解页面整体怎么串起来。
2. 再看 `ui/runtime_view.py`，它对应网页右侧的 runtime 学习区。
3. 接着看 `assistant_core/graph.py`，它对应真正的 LangGraph 节点和流转。
4. 最后看 `assistant_core/planner.py`、`assistant_core/retrieval.py`、`assistant_core/documents.py`，分别理解工具选择、检索和论文处理。
