# 论文阅读助手

这是一个适合新手练手的 LLM Demo：

- 上传一篇 PDF 或 Word（`.docx`）论文
- 自动解析文本并切分成 chunk
- 用 embedding 做相似度检索
- 把检索结果交给大模型回答问题
- 展示回答对应的原文证据

## 项目亮点

- 完整走通一条最小可用 RAG 链路
- 支持 OpenAI 兼容接口平台
- 兼容火山方舟的多模态 embedding 接口
- 适合写进 AI 实习简历，方便演示和讲解

## 项目结构

```text
app.py                # Streamlit 页面
paper_assistant.py    # 文档解析、切分、检索、回答逻辑
requirements.txt      # 依赖
.env.example          # 环境变量示例
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

把 `.env.example` 复制成 `.env`，填入你的平台配置：

```env
API_KEY=your_api_key_here
API_BASE_URL=https://your-provider-compatible-base-url
LLM_MODEL=your-chat-model-or-endpoint
EMBEDDING_MODEL=your-embedding-model-or-endpoint
```

如果你使用的是火山方舟，通常需要：

- `API_KEY`：火山方舟 API Key
- `API_BASE_URL`：例如 `https://ark.cn-beijing.volces.com/api/v3`
- `LLM_MODEL`：语言模型接入点，例如 `ep-...`
- `EMBEDDING_MODEL`：向量模型接入点，例如 `ep-...`

## 启动项目

```bash
streamlit run app.py
```

启动后，在浏览器中打开 Streamlit 给出的本地地址，上传论文后即可开始提问。

## 这个项目能学到什么

1. 文档解析
2. 文本切分
3. 向量化
4. 相似度检索
5. Prompt 组织
6. 基于证据回答

## 适合继续优化的方向

1. 支持多篇论文同时检索
2. 自动生成摘要、贡献点和实验总结
3. 增加引用高亮和答案出处卡片
4. 增加论文对比问答
5. 加一套简单评测集比较不同 chunk 参数效果
