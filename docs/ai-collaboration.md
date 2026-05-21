# AI 协作记录

## 概述
本项目在开发过程中使用 AI 协作，加速 MVP 交付与排错。

## 关键过程
- 建立 MVP 基础结构（app.py、rag_engine.py、requirements.txt）。
- 实现 PDF 解析、分块、本地 Chroma 向量库与 RAG 问答。
- 强制结构化摘要输出为固定 JSON 字段。
- 增加检索日志以支持可观测性。
- UI 切换为中文，并支持无文档问答。
- 适配 LangChain 模块化拆包后的兼容问题。
- 处理运行时问题：pypdf 缺失、Chroma persist 弃用、UTF-8 代理字符错误。

## 备注
所有修改均通过运行 Streamlit 应用并根据实际报错进行迭代验证。
