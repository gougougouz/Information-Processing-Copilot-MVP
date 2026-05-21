import hashlib
import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv

# 提前加载 .env，确保后续模块可读取环境变量
load_dotenv()

import streamlit as st

from rag_engine import RAGEngine

DATA_DIR = "data"

# 页面基础配置
st.set_page_config(page_title="本地研报 Copilot", layout="wide")


def _init_session_state() -> None:
    # 初始化 Streamlit 会话状态
    # 初始化会话状态，避免重复刷新时丢失上下文
    defaults = {
        "engine": None,
        "vectorstore": None,
        "summary": None,
        "summary_error": None,
        "retrieval_logs": [],
        "chat_history": [],
        "last_file_hash": None,
        "last_api_key_hash": None,
        "last_base_url": None,
        "last_processed_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _save_upload(uploaded_file: Any) -> str:
    # 保存上传文件到本地并返回路径
    # 将上传文件写入本地目录，便于后续解析
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = os.path.basename(uploaded_file.name)
    file_path = os.path.join(DATA_DIR, safe_name)
    with open(file_path, "wb") as handle:
        handle.write(uploaded_file.getvalue())
    return file_path


def _hash_bytes(data: bytes) -> str:
    # 计算文件内容哈希
    # 文件内容哈希，用于判断是否需要重新处理
    return hashlib.sha256(data).hexdigest()


def _hash_text(data: str) -> str:
    # 计算字符串哈希（用于 API Key）
    # API Key 哈希，避免在状态里保存明文
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _safe_text(text: str) -> str:
    # 处理可能的非法字符，避免控制台编码报错
    return text.encode("utf-8", "replace").decode("utf-8")


def _log_retrieval(chunks: List[Dict[str, Any]]) -> None:
    # 将检索命中信息打印到控制台
    # 控制台日志：打印命中的 chunk 片段与分数
    for entry in chunks:
        content = (entry.get("content") or "").replace("\n", " ")
        content = _safe_text(content)
        snippet = content[:300]
        print(
            "[RAG] rank={rank} score={score} similarity={similarity} source={source} page={page}\n{snippet}\n---".format(
                rank=entry.get("rank"),
                score=entry.get("score"),
                similarity=entry.get("similarity"),
                source=entry.get("source_file"),
                page=entry.get("page"),
                snippet=_safe_text(snippet),
            )
        )


def _render_retrieval_logs() -> None:
    # 在侧边栏渲染检索日志
    # 侧边栏展示最近检索日志，便于可观测性演示
    with st.sidebar.expander("检索日志", expanded=True):
        logs = st.session_state.retrieval_logs
        if not logs:
            st.caption("暂无检索日志。")
            return

        for entry in logs[-3:]:
            st.markdown(f"问题：{entry['question']}")
            for chunk in entry["chunks"]:
                score = chunk.get("score")
                similarity = chunk.get("similarity")
                source = chunk.get("source_file")
                page = chunk.get("page")
                st.markdown(
                    "片段 {rank} | 分数: {score} | 相似度: {similarity} | 来源: {source} | 页码: {page}".format(
                        rank=chunk.get("rank"),
                        score=f"{score:.4f}" if isinstance(score, float) else score,
                        similarity=f"{similarity:.4f}"
                        if isinstance(similarity, float)
                        else similarity,
                        source=source,
                        page=page,
                    )
                )
                st.code(_safe_text((chunk.get("content") or "")[:600]))


_init_session_state()

# 主标题
st.title("本地金融研报 Copilot (MVP)")

# 侧边栏：基础配置与文件上传
st.sidebar.header("配置")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    value="",
    placeholder="在此输入 API Key",
)
base_url = st.sidebar.text_input(
    "Base URL",
    value="",
    placeholder="https://api.openai.com/v1",
)
uploaded_file = st.sidebar.file_uploader("上传 PDF", type=["pdf"])

_render_retrieval_logs()

if uploaded_file is not None and (not api_key or not base_url):
    # 未填写关键信息时提示，防止无效调用
    st.sidebar.warning("请先输入 API Key 和 Base URL，再处理 PDF。")

if uploaded_file is not None and api_key and base_url:
    # 根据文件内容和配置变化判断是否需要重新处理
    file_bytes = uploaded_file.getvalue()
    file_hash = _hash_bytes(file_bytes)
    api_key_hash = _hash_text(api_key)

    needs_processing = (
        st.session_state.last_file_hash != file_hash
        or st.session_state.last_api_key_hash != api_key_hash
        or st.session_state.last_base_url != base_url
    )

    if needs_processing:
        # 记录最新状态，避免重复索引
        st.session_state.last_file_hash = file_hash
        st.session_state.last_api_key_hash = api_key_hash
        st.session_state.last_base_url = base_url

        # 初始化 RAG 引擎
        engine = RAGEngine(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            embedding_model=embedding_model,
        )
        st.session_state.engine = engine

        file_path = _save_upload(uploaded_file)

        with st.spinner("正在解析并建立索引..."):
            try:
                # 解析 -> 分块 -> 向量化 -> 结构化摘要
                engine.reset_vector_store()
                docs = engine.load_and_chunk_pdf(file_path)
                st.session_state.vectorstore = engine.build_vector_store(docs)
                st.session_state.summary = engine.generate_summary(docs)
                st.session_state.summary_error = None
                st.session_state.last_processed_at = time.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as exc:
                # 异常时清空结果并提示原因
                st.session_state.vectorstore = None
                st.session_state.summary = None
                st.session_state.summary_error = str(exc)

# 结构化摘要展示区
st.subheader("结构化摘要")
if st.session_state.summary_error:
    st.error(st.session_state.summary_error)
elif st.session_state.summary:
    st.json(st.session_state.summary)
elif st.session_state.last_processed_at:
    st.info("摘要为空或生成失败。")
else:
    st.info("请上传 PDF 生成结构化摘要。")

# 对话区：历史问答 + 输入框
st.subheader("研报问答")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("请输入问题")
if question:
    if not api_key or not base_url:
        st.warning("请先输入 API Key 和 Base URL。")
    else:
        api_key_hash = _hash_text(api_key)
        if (
            st.session_state.engine is None
            or st.session_state.last_api_key_hash != api_key_hash
            or st.session_state.last_base_url != base_url
        ):
            # 无需上传文件也可初始化引擎，用于通用问答
            st.session_state.engine = RAGEngine(
                api_key=api_key,
                base_url=base_url,
                model=model_name,
                embedding_model=embedding_model,
            )
            st.session_state.last_api_key_hash = api_key_hash
            st.session_state.last_base_url = base_url

        # 写入用户消息并展示
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在生成答案..."):
                try:
                    # 检索 + 生成回答，并记录命中日志
                    answer, chunks = st.session_state.engine.answer_question(
                        st.session_state.vectorstore, question
                    )
                    if chunks:
                        _log_retrieval(chunks)
                    st.session_state.retrieval_logs.append(
                        {"question": question, "chunks": chunks}
                    )
                except Exception as exc:
                    # 异常时直接反馈错误信息
                    answer = f"错误: {exc}"
            st.markdown(_safe_text(answer))

        # 写入助手回复
        st.session_state.chat_history.append(
            {"role": "assistant", "content": _safe_text(answer)}
        )
