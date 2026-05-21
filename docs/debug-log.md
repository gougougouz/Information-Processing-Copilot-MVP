# 排错记录

## 记录 1
- 日期：2026-05-21
- 问题：缺少 pypdf 依赖
- 现象："Failed to parse PDF: pypdf package not found"
- 原因：环境未安装依赖
- 解决：安装 pypdf 并写入 requirements.txt
- 状态：已解决

## 记录 2
- 日期：2026-05-21
- 问题：'utf-8' codec can't encode characters（surrogates）
- 现象：日志或前端渲染 PDF 内容时崩溃
- 原因：PDF 提取文本包含非法代理字符
- 解决：输出前统一做 UTF-8 replace 清洗
- 状态：已解决

## 记录 3
- 日期：2026-05-21
- 问题：'Chroma' object has no attribute 'persist'
- 现象：持久化向量库时报错
- 原因：版本差异导致 persist 方法不存在
- 解决：增加 hasattr 判断，存在才调用
- 状态：已解决
