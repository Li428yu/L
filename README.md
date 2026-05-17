# 论文阅读助手

这是一个适合练手和继续扩展的论文 RAG 项目。你上传论文之后，可以更快完成“读、问、比、记”这些事。

当前版本已经升级成真正的 `LangGraph agent`：

- 用显式 `ToolNode` 承载工具调用
- 用 `InMemorySaver` 提供多轮会话记忆
- 让 agent 在每一轮里自己决定是先列出论文、先检索证据，还是先生成摘要卡片

## 当前能力

- 上传 `PDF` 或 `DOCX` 论文
- 自动解析文本并切分成 chunk
- 使用 embedding 建立向量索引
- 支持多篇论文一起检索和问答
- 支持按论文筛选默认检索范围
- 支持多轮连续对话
- 支持展示本轮工具调用轨迹
- 支持展示最新检索证据
- 支持为单篇论文生成结构化摘要卡片

## 项目结构

```text
app.py                # Streamlit 页面和多轮对话交互
paper_assistant.py    # 文档解析、检索、工具定义、LangGraph agent
requirements.txt      # 依赖
.env.example          # 环境变量示例
```

## 安装依赖

推荐使用项目内虚拟环境：

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## 配置环境变量

复制 `.env.example` 为 `.env`，填入你自己的模型配置：

```env
API_KEY=your_api_key_here
API_BASE_URL=https://your-provider-compatible-base-url
LLM_MODEL=your-chat-model-or-endpoint
EMBEDDING_MODEL=your-embedding-model-or-endpoint
```

如果你使用的是 OpenAI 兼容平台，这四项一般就够了。

## 启动项目

```bash
.venv\Scripts\python -m streamlit run app.py
```

启动后你可以：

1. 一次上传一篇或多篇论文
2. 点击“建立索引”后再开始问答
3. 勾选默认检索范围
4. 连续追问，让 agent 记住上文
5. 查看它这一轮调用了哪些工具
6. 给单篇论文生成摘要卡片

## Agent 架构

当前问答链路不再是单个函数硬串，而是显式图结构：

1. `planner`
2. `tools`
3. `assistant`

其中：

- `planner` 会先把明显需要检索、列论文、生成卡片的问题路由到工具节点
- `tools` 由 `ToolNode` 执行真实工具
- `assistant` 再结合工具结果和会话记忆组织最终回答

目前内置的工具包括：

- `list_indexed_papers`
- `retrieve_paper_evidence`
- `generate_paper_digest_tool`

## 下一步适合继续扩展的方向

1. 增加查询改写节点，把口语追问改写成更稳的检索问题
2. 给检索结果增加重排和自检
3. 增加长期记忆，而不只是线程内短期记忆
4. 支持导出阅读笔记为 Markdown
5. 给证据片段做高亮和跳转
