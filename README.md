# 本地金融研报 Copilot (MVP)

一个本地 RAG 助手，用于处理金融研报 PDF。它会解析 PDF、分块并写入本地 Chroma 向量库，生成结构化 JSON 摘要，并在问答时返回带来源引用的答案。未上传文档时也可进行通用问答。

## 功能特性
- 本地 PDF 解析与分块
- 本地 Chroma 向量库持久化
- 结构化 JSON 摘要（固定 3 个字段）
- RAG 问答与来源引用
- 检索日志用于可观测性
- 无文档时支持通用问答

## 技术栈
- Python 3.10+
- Streamlit
- LangChain（模块化拆包）
- Chroma（本地）
- OpenAI 兼容 API

## 安装
```bash
pip install -r requirements.txt
```

## 启动
```bash
streamlit run app.py
```

## 配置
- API Key 和 Base URL 在 Streamlit 侧边栏手动输入。
- 可选环境变量（.env）：
  - OPENAI_MODEL（默认：gpt-4o-mini）
  - OPENAI_EMBEDDING_MODEL（默认：text-embedding-3-small）

## 使用流程
1. 在侧边栏输入 API Key 和 Base URL。
2. （可选）上传 PDF 以启用研报 RAG。
3. 在对话输入框提问。
4. 有检索内容时答案会带来源引用。

## 截图
关键截图说明放在 docs/运行截图/ 下。

## 关键文件
- app.py：Streamlit 界面、会话状态、聊天流程、日志
- rag_engine.py：PDF 解析、分块、向量检索、LLM 调用
- requirements.txt：依赖列表

## 测试样例
见 docs/test-cases.md。

## AI 协作记录
见 docs/ai-collaboration.md。

## 排错记录
见 docs/debug-log.md。
