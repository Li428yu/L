# 架构说明

## 总览

本项目定位是“论文阅读助手教学型 Demo”：一边提供面向用户的论文问答体验，一边把每轮回答背后的 RAG 链路、技术栈分层、检索证据和记忆过程可视化出来。

```text
frontend/                  React + Vite，面向用户的聊天式界面
backend/app/main.py         FastAPI API 入口
backend/app/indexer.py      上传后的后台索引任务
backend/app/document_processing.py
                            PDF/DOCX 解析、文档分析、自动 chunk 策略
backend/app/vector_store.py Chroma 持久化向量库
backend/app/agent.py        LangGraph RAG agent
backend/app/memory.py       用户画像、长期偏好、短期历史
backend/app/storage.py      SQLite 元数据存储
backend/app/evaluation.py   最小健康检查，默认不展示在用户界面
backend/app/eval_baselines.py
                            评测基线分层、默认基线和评测文档选择
data/                       上传文件、Chroma、SQLite，默认不入库
evals/                      固定问题评测集与 baselines.json
```

## 每一层负责什么

| 层 | 技术 | 职责 |
|---|---|---|
| 前端 | React + Vite | 聊天界面、上传入口、文档标签、原文证据、教学观察、阅读方式切换 |
| API 后端 | FastAPI | 文件上传、任务状态、问答 API、文档删除、评测入口、错误提示 |
| Agent harness | LangGraph | 把记忆、规划、检索、回答、写入记忆拆成可解释节点 |
| 向量数据库 | Chroma PersistentClient | 持久化 chunk、embedding、metadata，支持文档过滤和删除 |
| 元数据 | SQLite | 保存文档、任务、会话、消息、长期记忆 |
| 文档解析 | PyMuPDF + python-docx | 提取 PDF/DOCX 文本、页码、段落、表格文本 |

## 为什么这样选

### 前端：React + Vite

React 比 Streamlit 更适合做正式产品界面。这个项目需要左侧历史对话、中间聊天、右侧原文证据、教学观察面板、文档标签删除、模型切换和后续 PDF 高亮。Vite 启动快、配置轻，适合本地 MVP 和演示。

### 后端：FastAPI

FastAPI 适合把论文助手做成稳定 API，而不是页面脚本。它支持请求校验、文件上传、自动 API 文档、异步接口、后台任务，并且后续容易接 Celery/RQ、数据库权限、多用户登录和部署。

### Agent：LangGraph

LangGraph 的优势是把 agent 做成显式状态机，而不是把所有逻辑写在一个函数里。当前后端会用 `self.graph.stream(...)` 执行真实 LangGraph runtime，并通过 LangGraph custom stream 把状态和 token 传给前端。当前图结构是：

```text
memory -> planner -> answer
                 \-> retriever -> evidence_judge -> answer
                                               \-> retrieval_refiner -> retriever
answer -> verifier -> memory_writer
```

这让答辩时可以清楚说明 agent 的“感知、决策、工具调用、记忆、反馈”分别在哪里发生。前端不会直接展示生硬节点名，而是把它包装成“本轮运行链路”：

```text
接收问题 -> 读取记忆 -> 判断是否需要查文档
-> 可选检索证据 -> 证据裁判 -> 可选扩大范围重试
-> 生成/拒绝硬答 -> 引用核对 -> 更新记忆 -> 输出可观察信息
```

### 向量数据库：Chroma

当前阶段选择 Chroma，因为它本地部署轻、支持持久化、支持 metadata filter，足够证明项目不是内存 numpy 检索。它保存 chunk 文本、embedding 和文档 metadata，支持按文档筛选、删除和重启后保留索引。

### 检索管线：双路召回 + RRF

当前 `retriever` 不再只做向量 top-k。标准路径是：

```text
Dense 向量召回（Chroma cosine）
+ BM25 sparse 召回（本地倒排/IDF）
+ 少量结构化候选（仅字段、参考文献、对比、全文概览这类边界明确的问题）
-> RRF 融合
-> 按问题类型过滤
-> evidence_judge 裁判 direct/supporting/background/reject
```

当前项目不使用 rerank 模型，也不再为可靠性、局限、标题匹配、复合任务写复杂硬编码候选。普通问题走 Dense + BM25 + RRF；只有字段、参考文献、对比和全文概览这类结构边界清楚的问题才加入少量结构候选，后续只做通用过滤和证据裁判，不再调用外部 rerank 服务。

## Agent 能力映射

| 能力 | 当前实现 |
|---|---|
| 感知 | 读取用户问题、当前已上传文档、短期历史、长期用户画像 |
| 决策 | `planner` 判断问题是否需要检索，文档级问题是否需要分文档回答 |
| 工具调用 | `retriever` 调用 Dense 向量召回、BM25 sparse 召回和 RRF 融合来检索相关 chunk |
| 自适应重试 | `evidence_judge` 判定证据为空/弱/孤证时，路由到 `retrieval_refiner` 扩大检索范围 |
| 记忆 | `memory` 读取画像/偏好，`memory_writer` 写入长期记忆 |
| 反馈 | 前端展示引用、原文位置、教学观察、top-k、score、chunk 策略 |

当前属于 RAG agent MVP，不是强自主 agent。相比早期固定流水线，它已经支持按问题跳过检索、双路召回、RRF 融合、证据弱时重试、证据仍不足时拒绝硬答。后续可增加用户反馈节点和更细的 query rewrite。

## 向量数据库选型

| 方案 | 优势 | 局限 | 适合阶段 |
|---|---|---|---|
| Chroma | 本地轻量、持久化简单、metadata filter 够用 | 多用户权限和大规模生产能力较弱 | 当前本地 MVP |
| FAISS | 检索速度快、适合本地向量实验 | metadata、删除、持久化和服务化需要自己补 | 算法验证 |
| pgvector | 与 PostgreSQL 集成，适合权限、事务、多用户 | 部署比 Chroma 重，向量性能取决于索引设计 | 正式多用户系统 |
| Milvus | 面向大规模向量检索，性能和扩展性强 | 部署和运维复杂 | 海量文档生产环境 |

当前选择 Chroma；如果项目进入多用户阶段，建议迁移到 PostgreSQL + pgvector。

## 自动 chunk 策略

系统不让用户手动选择 chunk size 和 overlap，而是根据文档特征自动决定：

- 页数和总字符数
- 段落密度
- 平均每页字符数
- 中文、英文或中英文混合
- 公式和表格比例

选择结果保存到 `chunk_strategy`，包括：

- `chunk_size`
- `overlap`
- `splitter`
- `language`
- `page_count`
- `paragraph_count`
- `char_count`
- `reasons`

前端开发者视图会展示这些信息，用来解释“为什么这个文档这样切”。

## 教学观察设计

页面右侧分成两个标签：

| 标签 | 面向对象 | 展示内容 |
|---|---|---|
| 原文证据 | 普通阅读用户 | 当前点击引用对应的原文页码、段落、核心证据句、PDF/DOCX 原文件 |
| 教学观察 | 教学和答辩场景 | 本轮运行链路、RAG 检索、文档索引、Prompt 证据、记忆系统、技术栈地图 |

教学观察只解释当前这一轮问题，不会把固定评测集结果混在当前问答旁边。

## RAG 可证明链路

教学观察需要能证明系统真的在做 RAG，而不是直接把问题发给模型：

- 文档解析结果：页数、chunk 数、切分策略。
- 索引结果：embedding 模型、向量库类型、向量库记录数。
- 检索结果：top-k、返回证据数、score。
- 生成结果：最终 prompt 使用了哪些证据。
- 证据定位：点击引用可查看原文页码、段落、section。

## 已实现与待增强

| 模块 | 当前状态 | 后续增强 |
|---|---|---|
| 前后端分离 | 已实现 | 增加登录和多用户 |
| 文档上传 | 已实现 PDF/DOCX | 增加 OCR 和更多格式 |
| 向量数据库 | 已实现 Chroma | 生产阶段迁移 pgvector |
| 证据定位 | 已有页码/段落/PDF 页跳转 | 接 PDF.js 做坐标级高亮 |
| 模型切换 | 已有阅读方式预设 | 增加 provider、成本、失败提示说明 |
| 记忆 | 已有画像/偏好/短期历史 | 增加会话摘要和跨 500 轮压缩记忆 |
| 评测 | 后端只保留基础健康检查，前端默认不展示 | 观察证据链路是否可用、引用是否可点开 |
| 异步任务 | FastAPI BackgroundTasks | 长任务迁移 Celery/RQ |
| 错误处理 | 已有主要提示和 fallback | 增加 OCR、限流、向量库连接失败细分 |
