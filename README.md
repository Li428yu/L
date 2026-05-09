# 论文阅读助手

这是一个适合练手和继续扩展的 RAG 小项目，核心目标是让你把论文上传进去之后，能更快完成“读、问、比、记”这几件事。

当前版本已经支持：

- 上传 `PDF` 或 `DOCX` 论文
- 自动解析文本并切分成 chunk
- 使用 embedding 建立向量索引
- 支持多篇论文一起检索和问答
- 支持按论文筛选检索范围
- 为单篇论文生成结构化摘要卡片
- 展示回答对应的原文证据

## 项目结构

```text
app.py                # Streamlit 页面
paper_assistant.py    # 文档解析、切分、检索、摘要与问答逻辑
requirements.txt      # 依赖
.env.example          # 环境变量示例
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置环境变量

复制 `.env.example` 为 `.env`，填入你自己的模型配置：

```env
API_KEY=your_api_key_here
API_BASE_URL=https://your-provider-compatible-base-url
LLM_MODEL=your-chat-model-or-endpoint
EMBEDDING_MODEL=your-embedding-model-or-endpoint
```

如果你使用的是 OpenAI 兼容平台，一般需要这四项就够了。

## 启动项目

```bash
streamlit run app.py
```

启动后你可以：

1. 一次上传一篇或多篇论文
2. 点击“建立索引”后再开始问答
3. 勾选指定论文做定向检索
4. 给单篇论文生成摘要卡片

## 这版新增了什么

### 1. 多论文联合检索

现在不再限制单篇论文。你可以把几篇相关工作一起传进去，直接问：

- 这几篇论文的核心差异是什么？
- 哪一篇实验更充分？
- 方法设计上最大的区别在哪里？

### 2. 手动建立 / 更新索引

旧版本只要页面一刷新，就可能重复走一遍 embedding。现在改成：

- 上传文件后由你决定何时建立索引
- 改了 chunk 参数或换了文件时会提示索引已过期
- 避免无意义的重复消耗

### 3. 论文摘要卡片

你可以对单篇论文生成结构化阅读卡片，输出包括：

- 一句话总结
- 研究问题
- 核心方法
- 主要贡献
- 实验与结果
- 局限性
- 推荐追问

## 适合继续优化的方向

这个项目还很适合继续往下做，比如：

1. 给答案里的引用做高亮和跳转
2. 自动抽取摘要、方法、实验、结论等章节
3. 支持阅读笔记导出为 Markdown
4. 增加多轮对话记忆
5. 为不同 chunk 参数做效果评测
