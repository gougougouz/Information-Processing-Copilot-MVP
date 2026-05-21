# 排错记录

## 记录 1
- 问题：'utf-8' codec can't encode characters（surrogates）
- 现象：日志或前端渲染 PDF 内容时崩溃
- 原因：PDF 提取文本包含非法代理字符
- 解决：输出前统一做 UTF-8 replace 清洗

## 记录 2
- 问题：'Chroma' object has no attribute 'persist'
- 现象：持久化向量库时报错
- 原因：版本差异导致 persist 方法不存在
- 解决：增加 hasattr 判断，存在才调用

