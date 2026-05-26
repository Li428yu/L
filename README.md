# 论文阅读助手

这是一个面向教学展示的论文阅读助手 Demo。它不只让用户和论文对话，也会把每一轮回答背后的 RAG 链路、技术栈分层、证据来源和记忆过程展示出来，帮助初学者看懂“论文阅读 Agent”内部发生了什么。

当前主线是前后端分离项目：

- 前端：React + Vite
- 后端：FastAPI
- Agent harness：LangGraph
- 向量数据库：Chroma PersistentClient
- 元数据存储：SQLite
- 文档解析：PyMuPDF + python-docx

主项目入口是 `backend/` 和 `frontend/`。

## 当前能力

- 用户通过聊天输入框左侧的 `+` 上传 PDF 或 DOCX。
- 上传后自动解析、自动选择 chunk 参数、生成 embedding、写入 Chroma。
- 输入框上方展示已上传文档，文档旁边可直接删除。
- 用户提问时默认针对所有已准备好的文档回答；多篇文档的总结类问题会按文档分开回答。
- 回答中的证据编号可点击，右侧展示原文段落、页码、section，并支持打开 PDF 对应页或下载 DOCX。
- 阅读用户可以只看聊天、文档和原文证据；教学讲解时再打开“教学观察”。
- 右侧支持“原文证据 / 教学观察”切换：原文证据用于阅读，教学观察用于讲清楚本轮运行链路。
- 支持“日常阅读 / 精读模式 / 快速浏览”三种阅读方式。
- 记忆结构包含短期对话历史和长期用户画像/偏好。

## 教学观察面板

右侧“教学观察”不是评测集，也不会额外替用户问很多问题。它只解释当前这一轮问题是怎么被处理的：

1. 前端把问题发送给 FastAPI。
2. LangGraph 读取记忆，判断是否需要检索文档。
3. 系统使用 Dense 向量召回和 BM25 sparse 召回分别找候选 chunk，再用 RRF 融合排序。
4. 后端把证据、用户问题、最近对话和记忆组合进 Prompt。
5. 模型生成回答，前端把引用变成可点击的原文证据。
6. 系统更新短期历史和长期用户偏好。

教学观察面板会展示：

- 本轮运行链路。
- RAG 检索过程、top-k、score、检索管线、RRF 排序方式、向量库记录数。
- 每篇文档的页数、chunk 数、chunk_size、overlap、切分理由。
- 最终用于回答的证据。
- 记忆系统当前用到的信息。
- React / FastAPI / LangGraph / Chroma / SQLite 等技术栈各自负责什么。

## 安装

进入项目目录：

```powershell
cd E:\ai1\ai-ai-demo
```

推荐一键安装：

```powershell
.\setup_project.ps1
```

如果 PowerShell 脚本被拦截，可以运行：

```powershell
.\setup_project.bat
```

手动安装方式：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd frontend
npm.cmd install
```

复制 `.env.example` 为 `.env`，填入模型服务配置：

```env
API_KEY=your_api_key_here
API_BASE_URL=https://your-provider-compatible-base-url
LLM_MODEL=your-chat-model
EMBEDDING_MODEL=your-embedding-model
```

普通界面不会展示 API endpoint 或密钥。

## 启动

开两个终端：

```powershell
.\start_backend.ps1
```

```powershell
.\start_frontend.ps1
```

然后访问：

```text
http://127.0.0.1:5173
```

后端 API 文档：

```text
http://127.0.0.1:8000/docs
```

## RAG 链路

1. 用户上传 PDF 或 DOCX。
2. 后端计算 `file_hash` 并保存原文件。
3. 后台任务读取文档文本。
4. 系统根据页数、段落密度、语言、公式/表格比例自动选择 chunk size、overlap 和 splitter。
5. 每个 chunk 保存 `document_id`、`paper_name`、`page`、`section`、`source`、`file_hash`、`char_start`、`char_end`。
6. 调用 embedding 模型生成向量；如果明确选择 `local-hash-embedding-v1`，则使用可复现的本地哈希向量作为本地基线。若请求远程 embedding 但服务不可用，系统会记录为 fallback，相关评测不可作为可信横向对比。
7. chunk、embedding、metadata 写入 Chroma 持久化向量库。
8. 用户提问时，LangGraph 读取记忆，判断是否需要检索。
9. retriever 节点同时运行 Dense 向量召回和 BM25 sparse 召回，并只合并少量结构化候选，例如字段、参考文献和文档概览。
10. 系统用 RRF 融合多路候选，直接按 RRF 融合分排序，不使用 rerank 模型。
11. answer 节点把证据、记忆和短期历史组成最终 prompt，生成回答。
12. 前端展示回答、引用、原文证据，并在教学观察面板展示 chunk、score、top-k、检索管线、排序方式和最终使用证据。

## 为什么同时有 LangChain 和 LangGraph

项目使用 LangGraph 做 agent 流程编排，并由 `self.graph.stream(...)` 执行真实 runtime。当前 `StateGraph` 不是固定直线流程，而是：

```text
memory -> planner -> answer
                 \-> retriever -> evidence_judge -> answer
                                               \-> retrieval_refiner -> retriever
answer -> verifier -> memory_writer
```

也就是说，普通使用说明类问题会跳过文档检索；文档问答会先检索和裁判证据；如果证据为空、过弱或只有一条孤证，图会扩大检索范围再跑一轮；重试后仍缺少直接证据时，回答节点会拒绝硬编。

项目没有使用完整 LangChain 链式框架，但使用了 LangChain 的消息类型和模型生态适配能力，例如 `SystemMessage`、`HumanMessage`。因此依赖里出现 `langgraph` 和 `langchain-openai` 是合理的：前者负责编排，后者负责模型消息和供应商适配。

## 教学实验与评测

固定问题评测集仍保留在后端和 `evals/` 目录中，用于开发阶段做回归测试；它不再出现在普通问答页面里，避免用户误以为系统在回答当前问题时额外跑了多个问题。

评测默认基线由 `evals/baselines.json` 管理。当前默认只跑 `pdf_gold_current`，也就是带 `expected_chunk_ids` 的 PDF gold 基线；缺失本地 DOCX 文档的网络程序设计和 PDF/DOCX 混合基线已标记为 paused，避免不可运行 case 污染平均分。

每轮评测会额外输出分级报告：单条 case 有 `result_status`、`failure_categories` 和 `grading_reasons`，整轮结果有 `result_status_counts`、`failure_category_counts`、`evaluation_trustworthy` 和 `trust_gate_status`。如果发生 embedding fallback，本轮会被标为 `not_comparable`，只能用于诊断，不能作为可信基线或横向对比成绩。
