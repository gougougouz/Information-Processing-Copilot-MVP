import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class RAGEngine:
    # 初始化 RAG 引擎配置
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        persist_dir: str = "chroma_db",
        collection_name: str = "research_reports",
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        request_timeout: int = 30,
    ) -> None:
        # 基础配置：模型、向量库目录、超时等
        self.api_key = api_key
        self.base_url = base_url or None
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model = model
        self.embedding_model = embedding_model
        self.request_timeout = request_timeout

    # 清理本地向量库目录
    def reset_vector_store(self) -> None:
        # 重新处理文件时清理旧向量库，避免混入历史数据
        if os.path.isdir(self.persist_dir):
            shutil.rmtree(self.persist_dir, ignore_errors=True)

    # 解析 PDF 并按规则分块
    def load_and_chunk_pdf(self, file_path: str) -> List[Any]:
        try:
            # 读取 PDF 并按页解析为 Document 列表
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse PDF: {exc}") from exc

        for doc in docs:
            doc.page_content = self._sanitize_text(doc.page_content or "")

        # 研报分块：控制上下文长度并保留重叠，提升语义连续性
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        source_file = os.path.basename(file_path)
        for doc in chunks:
            # 统一补充来源文件与页码，便于后续溯源
            metadata = doc.metadata or {}
            metadata["source_file"] = source_file
            if "page" in metadata:
                metadata["page"] = self._safe_page_index(metadata.get("page"))
            doc.metadata = metadata

        return chunks

    # 将分块写入本地 Chroma 向量库（langchain-chroma）
    def build_vector_store(self, docs: List[Any]) -> Chroma:
        # 基于 OpenAI Embeddings 构建本地 Chroma 向量库
        embeddings = self._make_embeddings()
        vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
        )
        if docs:
            # 将分块文本写入向量库
            vectorstore.add_documents(docs)
        # 持久化到本地磁盘，方便重启后复用（部分版本无 persist）
        if hasattr(vectorstore, "persist"):
            vectorstore.persist()
        return vectorstore

    # 生成固定 JSON 结构的研报摘要
    def generate_summary(self, docs: List[Any]) -> Dict[str, Any]:
        # 将文档内容截断汇总为固定长度输入，避免超长
        content = self._prepare_summary_text(docs)
        if not content.strip():
            return {}

        llm = self._make_chat_model()
        # 强制模型输出固定结构 JSON，便于前端结构化展示
        system_prompt = (
            "You are a financial research analyst. "
            "Return ONLY valid JSON with exactly three keys: "
            "\"核心观点\", \"利好因素\", \"潜在风险\". "
            "Do not add any extra text."
        )
        user_prompt = f"Document content:\n{content}"

        try:
            # 调用大模型生成结构化摘要
            response = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            raise RuntimeError(f"LLM request failed while generating summary: {exc}") from exc

        return self._parse_json_response(response.content)

    # 基于检索上下文回答问题并返回日志
    def answer_question(
        self, vectorstore: Optional[Chroma], question: str, k: int = 4
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not question.strip():
            return "请提供问题。", []

        results = []
        if vectorstore is not None:
            try:
                # 向量检索：返回 (Document, score)
                results = vectorstore.similarity_search_with_score(question, k=k)
            except Exception as exc:
                raise RuntimeError(f"Retrieval failed: {exc}") from exc

        # 记录检索命中用于可观测性展示，供前端侧边栏输出
        logs: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for idx, (doc, score) in enumerate(results, start=1):
            content = self._sanitize_text((doc.page_content or "").strip())
            metadata = doc.metadata or {}
            source_file = metadata.get("source_file") or os.path.basename(
                metadata.get("source", "")
            )
            page = metadata.get("page")
            score_value = self._safe_float(score)
            similarity = None
            if score_value is not None:
                try:
                    # 将距离分数转换为近似相似度，便于理解
                    similarity = 1.0 / (1.0 + score_value)
                except Exception:
                    similarity = None

            logs.append(
                {
                    "rank": idx,
                    "content": content,
                    "score": score_value,
                    "similarity": similarity,
                    "source_file": source_file,
                    "page": page,
                }
            )

            header = f"[{idx}] 来源文件: {source_file}, 页面: {page}"
            context_blocks.append(f"{header}\n{content}")

        llm = self._make_chat_model()
        # 优先使用检索内容；资料不足时允许基于常识补充并明确说明
        system_prompt = (
            "You are a helpful financial assistant. "
            "Use the provided context if available and cite it. "
            "If context is missing or insufficient, answer using general knowledge, "
            "and clearly state that those parts are not from the documents. "
            "Respond in Chinese."
        )
        context_text = os.linesep.join(context_blocks) if context_blocks else "（无）"
        user_prompt = f"问题: {question}\n\n可参考资料:\n{context_text}"

        try:
            # 用检索片段作为上下文生成答案
            response = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            raise RuntimeError(f"LLM request failed while answering question: {exc}") from exc

        answer = str(response.content).strip()
        if results:
            empty_reason = "未检索到相关文档"
        else:
            empty_reason = "未上传文档或未建立索引"
        sources = self._format_sources(logs, empty_reason)
        if sources:
            # 回答尾部追加溯源信息
            answer = f"{answer}\n\n{sources}"

        return answer, logs

    # 构建聊天模型（OpenAI 兼容接口）
    def _make_chat_model(self) -> ChatOpenAI:
        # OpenAI 兼容接口初始化，适配 base_url/api_key
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0.2,
            "timeout": self.request_timeout,
            "max_retries": 2,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        try:
            return ChatOpenAI(**kwargs)
        except TypeError:
            # 兼容旧版参数命名
            legacy_kwargs = {
                "openai_api_key": self.api_key,
                "openai_api_base": self.base_url,
                "model": self.model,
                "temperature": 0.2,
                "timeout": self.request_timeout,
                "max_retries": 2,
            }
            return ChatOpenAI(**legacy_kwargs)

    # 构建向量嵌入模型
    def _make_embeddings(self) -> OpenAIEmbeddings:
        # Embeddings 用于向量检索
        kwargs: Dict[str, Any] = {
            "model": self.embedding_model,
            "timeout": self.request_timeout,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url

        try:
            # 新版 OpenAIEmbeddings 参数
            return OpenAIEmbeddings(**kwargs)
        except TypeError:
            # 兼容旧版参数命名
            legacy_kwargs = {
                "openai_api_key": self.api_key,
                "openai_api_base": self.base_url,
                "model": self.embedding_model,
                "timeout": self.request_timeout,
            }
            return OpenAIEmbeddings(**legacy_kwargs)

    @staticmethod
    # 拼接并截断摘要输入文本
    def _prepare_summary_text(docs: List[Any], max_chars: int = 12000) -> str:
        # 将多页内容拼接并截断到 max_chars，避免超出上下文限制
        parts: List[str] = []
        total = 0
        for doc in docs:
            text = (doc.page_content or "").strip()
            if not text:
                continue
            remaining = max_chars - total
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining]
            parts.append(text)
            total += len(text)
        return "\n\n".join(parts)

    @staticmethod
    # 解析模型返回的 JSON（含兜底策略）
    def _parse_json_response(content: Any) -> Dict[str, Any]:
        if isinstance(content, dict):
            return content
        text = str(content).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 兜底：从文本中提取可能的 JSON 片段
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        raise RuntimeError("Model returned invalid JSON. Please try again.")

    @staticmethod
    # 页码安全转换
    def _safe_page_index(value: Any) -> Any:
        # 将 0-based 页码转换为 1-based，方便阅读
        try:
            return int(value) + 1
        except Exception:
            return value

    @staticmethod
    # 分数安全转换
    def _safe_float(value: Any) -> Optional[float]:
        # 分数解析失败时返回 None，避免影响展示
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _sanitize_text(text: str) -> str:
        return text.encode("utf-8", "replace").decode("utf-8")

    @staticmethod
    # 将检索日志整理为溯源文本
    def _format_sources(logs: List[Dict[str, Any]], empty_reason: str) -> str:
        seen = set()
        lines: List[str] = []
        for entry in logs:
            source = entry.get("source_file") or "未知"
            page = entry.get("page") if entry.get("page") is not None else "未知"
            key = (source, page)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"[来源文件: {source}, 页面: {page}]")
        if not lines:
            return f"来源:\n无（{empty_reason}）"
        # 统一在回答末尾输出溯源信息
        return "来源:\n" + "\n".join(lines)
