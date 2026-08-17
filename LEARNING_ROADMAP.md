# 🗺️ Vietnamese Legal Assistant - Lộ Trình Học Tập 14 Ngày

## 📚 Tổng Quan

Dự án này xây dựng một **RAG Chatbot pháp lý tiếng Việt** với:
- FastAPI backend
- RAG (Retrieval-Augmented Generation)
- Agentic RAG với LangGraph
- Memory & Context Engineering
- Production optimizations

**Thời gian**: 14 ngày (mỗi ngày ~2-4 giờ)

---

## 🎯 Prerequisites

Trước khi bắt đầu, đảm bảo bạn có:

### Kiến thức nền tảng
- [ ] Python intermediate (decorators, async, dataclasses)
- [ ] REST API basics (HTTP methods, JSON)
- [ ] Git basics
- [ ] Biết vector, embedding là gì (sẽ học ở Buổi 4)

### Công cụ cần cài
```bash
# Python environment
conda create -n legal-assistant python==3.10
conda activate legal-assistant

# Dependencies cơ bản
pip install fastapi uvicorn pydantic pydantic-settings openai qdrant-client python-dotenv

# Dev tools
pip install pytest pytest-mock httpx
```

---

## 📅 NGÀY 1-2: FastAPI Setup & Hello World

### Mục tiêu
- [x] Tạo project structure
- [x] FastAPI app chạy được
- [x] Health check endpoint
- [x] Static file serving

### Cấu trúc thư mục
```
legal-assistant/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI entry point
│   └── config.py         # Settings từ .env
├── .env
├── .env.example
├── requirements.txt
└── README.md
```

### Bài tập
1. Tạo `config.py` dùng `pydantic-settings`:
```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Legal Assistant"
    openai_api_key: str = ""
    
    class Config:
        env_file = ".env"
```

2. Tạo `main.py` với:
   - GET `/` → trả file HTML (copy `app/static/chat.html` từ repo mẫu)
   - GET `/health` → health check đơn giản
   - POST `/chat` → stub trả "Xin chào, tôi là trợ lý pháp lý"

3. Chạy: `uvicorn app.main:app --reload`

### Kết quả mong đợi
```
curl http://localhost:8000/health
# {"status": "ok"}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Xin chào"}'
# "Xin chào, tôi là trợ lý pháp lý"
```

---

## 📅 NGÀY 3-4: LLM API Integration

### Mục tiêu
- [x] Gọi OpenAI Chat Completions API
- [x] Streaming response
- [x] Structured output với Pydantic
- [x] Error handling & retry

### Kiến thức học được
1. **OpenAI Chat Completions API**
   - Messages format: system, user, assistant
   - Parameters: model, temperature, max_tokens

2. **Streaming**
   - Server-Sent Events (SSE)
   - Generator pattern trong Python

3. **Structured Output**
   - `response_format` parameter
   - Pydantic schema validation

### Cấu trúc thư mục mới
```
app/
├── llm/
│   ├── __init__.py
│   ├── client.py      # OpenAI client factory
│   ├── completion.py # chat(), chat_stream(), chat_parsed()
│   └── params.py     # GenerationParams dataclass
├── prompts/
│   └── templates.py  # System prompt
└── schemas/
    └── domain.py     # Pydantic models cho structured output
```

### Bài tập

#### 1. Tạo `GenerationParams`
```python
# app/llm/params.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationParams:
    temperature: float = 0.2
    max_tokens: int = 800
    top_p: float = 1.0
```

#### 2. Tạo `client.py` - OpenAI client
```python
# app/llm/client.py
from openai import OpenAI
from app.config import settings

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client
```

#### 3. Tạo `completion.py` - 3 hàm chính
```python
# app/llm/completion.py
from openai import OpenAI

def chat(messages: list[dict], params: GenerationParams) -> str:
    """Gọi chat completion, trả text"""
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
    )
    return response.choices[0].message.content

def chat_stream(messages: list[dict], params: GenerationParams):
    """Streaming - yield từng token"""
    client = get_client()
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
        **params.__dict__
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def chat_parsed(messages: list[dict], schema: type):
    """Structured output - trả object theo Pydantic schema"""
    client = get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format=schema,
    )
    return response.choices[0].message.parsed
```

#### 4. Tạo system prompt
```python
# app/prompts/templates.py
LEGAL_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật doanh nghiệp Việt Nam.

Quy tắc:
- Trả lời chính xác, ngắn gọn, bằng tiếng Việt
- Khi có tài liệu tham khảo, trả lời dựa trên tài liệu đó
- Khi không có tài liệu, nói rõ và khuyến khích kiểm chứng
"""

def build_messages(question: str, context: str = "") -> list[dict]:
    messages = [
        {"role": "system", "content": LEGAL_SYSTEM_PROMPT}
    ]
    if context:
        messages[0]["content"] += f"\n\n--- TÀI LIỆU THAM KHẢO ---\n{context}"
    messages.append({"role": "user", "content": question})
    return messages
```

#### 5. Cập nhật `/chat` endpoint
```python
# Trong main.py hoặc routes_chat.py
@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    messages = build_messages(req.question)
    answer = chat(messages, GenerationParams())
    return {"answer": answer}
```

### Kết quả mong đợi
```bash
# Non-streaming
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Công ty TNHH là gì?"}'

# Streaming  
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Điều kiện thành lập công ty?"}'
```

---

## 📅 NGÀY 5-6: Local LLM (Ollama/vLLM)

### Mục tiêu
- [x] Chạy model local (Ollama)
- [x] Switch giữa OpenAI và local backend
- [x] Không đổi code, chỉ đổi config

### Kiến thức học được
1. **OpenAI-compatible API**
   - Ollama và vLLM đều expose endpoint `/v1/chat/completions`
   - Dùng chung OpenAI client, chỉ đổi `base_url`

2. **Environment-based configuration**
   - Backend selection qua env var
   - Key rotation cho multi-key

### Bài tập

#### 1. Tạo backend configuration
```python
# app/llm/backends.py
from dataclasses import dataclass

@dataclass
class Backend:
    name: str
    base_url: str  # "" = OpenAI Cloud
    requires_key: bool

BACKENDS = {
    "openai": Backend("openai", "", True),
    "ollama": Backend("ollama", "http://localhost:11434/v1", False),
    "vllm": Backend("vllm", "http://localhost:8000/v1", False),
}
```

#### 2. Cập nhật `client.py`
```python
# app/llm/client.py
from app.config import settings
from app.llm.backends import BACKENDS

def get_client() -> OpenAI:
    backend = BACKENDS[settings.llm_backend]
    api_key = "dummy" if not backend.requires_key else settings.openai_api_key
    return OpenAI(
        api_key=api_key,
        base_url=backend.base_url if backend.base_url else None
    )
```

#### 3. Cài Ollama và test
```bash
# Cài Ollama (macOS/Linux/Windows)
# Pull model
ollama pull llama3.2:3b

# Test nhanh
ollama run llama3.2:3b "Xin chào"

# Trong .env:
LLM_BACKEND=ollama
LLM_MODEL=llama3.2:3b
```

---

## 📅 NGÀY 7-8: Embeddings & Vector Database

### Mục tiêu
- [x] Tạo embeddings cho text (OpenAI)
- [x] Lưu trữ vectors trong Qdrant
- [x] Semantic search

### Kiến thức học được
1. **Embeddings**
   - Text → vector (1536 dimensions)
   - Cosine similarity
   - OpenAI `text-embedding-3-small`

2. **Vector Database**
   - Qdrant (HNSW index)
   - CRUD operations
   - Filtering

### Cấu trúc thư mục mới
```
app/retrieval/
├── __init__.py
├── embeddings.py  # OpenAI embeddings
├── vectorstore.py # Qdrant operations
├── loader.py      # Load documents
├── chunking.py    # Split text
└── retriever.py   # Search pipeline
```

### Bài tập

#### 1. Embeddings
```python
# app/retrieval/embeddings.py
from openai import OpenAI

_client = OpenAI()

def embed_text(text: str) -> list[float]:
    response = _client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def embed_batch(texts: list[str]) -> list[list[float]]:
    response = _client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
```

#### 2. Vector Store (Qdrant)
```python
# app/retrieval/vectorstore.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

_client = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(location=":memory:")  # hoặc url="http://localhost:6333"
    return _client

def ensure_collection(name: str, dim: int):
    client = get_client()
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )

def add_documents(ids: list[str], vectors: list[list[float]], texts: list[str]):
    client = get_client()
    points = [
        PointStruct(id=i, vector=v, payload={"text": t})
        for i, v, t in zip(ids, vectors, texts)
    ]
    client.upsert(collection_name="legal_docs", points=points)

def search(query_vector: list[float], top_k: int = 5) -> list[dict]:
    client = get_client()
    results = client.search(
        collection_name="legal_docs",
        query_vector=query_vector,
        limit=top_k
    )
    return [{"text": h.payload["text"], "score": h.score} for h in results]
```

#### 3. Test nhanh
```python
# scripts/test_retrieval.py
from app.retrieval.embeddings import embed_text
from app.retrieval.vectorstore import ensure_collection, add_documents, search

# Setup
ensure_collection("legal_docs", 1536)

# Index
docs = [
    "Điều 46. Công ty trách nhiệm hữu hạn hai thành viên...",
    "Điều 47. Công ty cổ phần...",
]
vectors = [embed_text(d) for d in docs]
add_documents(["d1", "d2"], vectors, docs)

# Search
query = "điều kiện thành lập công ty"
q_vec = embed_text(query)
results = search(q_vec, top_k=2)
print(results)
```

---

## 📅 NGÀY 9-10: RAG Pipeline Hoàn Chỉnh

### Mục tiêu
- [x] Load & chunk documents
- [x] Full RAG: retrieve → augment → generate
- [x] Ingestion script

### Kiến thức học được
1. **Document Loading**
   - TXT, Markdown, PDF parsing
   - Metadata extraction

2. **Text Chunking**
   - Recursive character splitting
   - Overlap để giữ context

3. **RAG Pattern**
   - Retrieve: semantic search
   - Augment: chèn context vào prompt
   - Generate: LLM trả lời

### Bài tập

#### 1. Document Loader
```python
# app/retrieval/loader.py
from pathlib import Path

def load_document(path: Path) -> tuple[str, dict]:
    """Load document, return (text, metadata)"""
    text = path.read_text(encoding="utf-8")
    metadata = {
        "source": path.name,
        "type": path.suffix
    }
    return text, metadata

def load_directory(dir_path: str) -> list[tuple[str, dict]]:
    docs = []
    for path in Path(dir_path).rglob("*.md"):
        text, meta = load_document(path)
        docs.append((text, meta))
    return docs
```

#### 2. Text Chunker
```python
# app/retrieval/chunking.py
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Recursive character splitter"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    current = ""
    
    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += "\n\n" + para
        else:
            if current:
                chunks.append(current)
            current = para
    
    if current:
        chunks.append(current)
    
    return chunks
```

#### 3. Retriever
```python
# app/retrieval/retriever.py
from app.retrieval.embeddings import embed_text
from app.retrieval.vectorstore import search

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search"""
    query_vector = embed_text(query)
    return search(query_vector, top_k=top_k)
```

#### 4. Cập nhật Pipeline
```python
# app/pipeline.py
from app.retrieval.retriever import retrieve
from app.prompts.templates import build_messages
from app.llm.completion import chat

def answer(question: str) -> str:
    # Retrieve
    chunks = retrieve(question)
    context = "\n\n".join(c["text"] for c in chunks)
    
    # Build messages with context
    messages = build_messages(question, context)
    
    # Generate
    return chat(messages, GenerationParams())
```

#### 5. Ingestion Script
```python
# scripts/ingest.py
from app.retrieval.loader import load_directory
from app.retrieval.chunking import chunk_text
from app.retrieval.embeddings import embed_batch
from app.retrieval.vectorstore import ensure_collection, add_documents

def ingest(source_dir: str):
    docs = load_directory(source_dir)
    all_chunks = []
    all_ids = []
    
    for i, (text, meta) in enumerate(docs):
        chunks = chunk_text(text)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{meta['source']}_{j}")
    
    vectors = embed_batch(all_chunks)
    ensure_collection("legal_docs", 1536)
    add_documents(all_ids, vectors, all_chunks)
    print(f"Indexed {len(all_chunks)} chunks")

if __name__ == "__main__":
    ingest("./data/legal_docs")
```

---

## 📅 NGÀY 11-12: Agentic RAG với LangGraph

### Mục tiêu
- [x] Query decomposition
- [x] Document grading
- [x] Web search fallback
- [x] LangGraph orchestration

### Kiến thức học được
1. **LangGraph**
   - StateGraph
   - Nodes & Edges
   - Conditional routing

2. **Agentic Patterns**
   - CRAG (Corrective RAG)
   - Query decomposition
   - Grading & routing

### Cấu trúc thư mục mới
```
app/agent/
├── __init__.py
├── state.py      # GraphState dataclass
├── nodes.py      # Node functions
├── graph.py      # Graph definition
└── tools_web.py  # Web search tool
```

### Bài tập

#### 1. Define State
```python
# app/agent/state.py
from dataclasses import dataclass

@dataclass
class GraphState:
    question: str
    sub_questions: list[str] = None
    documents: list[dict] = None
    generation: str = ""
    web_search_used: bool = False
```

#### 2. Create Nodes
```python
# app/agent/nodes.py
from app.llm.completion import chat, chat_parsed
from pydantic import BaseModel

class SubQuestions(BaseModel):
    questions: list[str]

def decompose(state: GraphState) -> dict:
    """Split complex question into sub-questions"""
    if len(state.question) < 50:
        return {"sub_questions": [state.question]}
    
    messages = [
        {"role": "system", "content": "Chia câu hỏi thành 2-4 câu hỏi con"},
        {"role": "user", "content": state.question}
    ]
    result = chat_parsed(messages, SubQuestions)
    return {"sub_questions": result.questions}

def retrieve(state: GraphState) -> dict:
    """Retrieve documents for each sub-question"""
    from app.retrieval.retriever import retrieve as do_retrieve
    
    all_docs = []
    for sq in state.sub_questions:
        docs = do_retrieve(sq)
        all_docs.extend(docs)
    return {"documents": all_docs}

def grade(state: GraphState) -> dict:
    """Filter to relevant documents only"""
    # Simple: keep top 5 by score
    docs = sorted(state.documents, key=lambda d: d.get("score", 0), reverse=True)[:5]
    return {"documents": docs}

def generate(state: GraphState) -> dict:
    """Generate answer from documents"""
    context = "\n\n".join(d["text"] for d in state.documents)
    messages = build_messages(state.question, context)
    answer = chat(messages, GenerationParams(temperature=0.1))
    return {"generation": answer}
```

#### 3. Build Graph
```python
# app/agent/graph.py
from langgraph.graph import StateGraph, END

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("decompose", decompose)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("generate", generate)
    
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def run_agent(question: str) -> dict:
    graph = build_graph()
    return graph.invoke({"question": question})
```

---

## 📅 NGÀY 13-14: Evaluation & Production Optimization

### Mục tiêu
- [x] LLM-as-Judge evaluation
- [x] Semantic caching
- [x] Model routing
- [x] Guardrails

### Bài tập

#### 1. LLM-as-Judge
```python
# app/eval/judge.py
from pydantic import BaseModel

class JudgeScore(BaseModel):
    accuracy: int  # 1-5
    completeness: int  # 1-5
    groundedness: int  # 1-5

def judge(question: str, answer: str, context: str) -> JudgeScore:
    messages = [
        {"role": "system", "content": "Chấm điểm câu trả lời 1-5"},
        {"role": "user", "content": f"Q: {question}\nA: {answer}\nCtx: {context}"}
    ]
    return chat_parsed(messages, JudgeScore)
```

#### 2. Semantic Cache
```python
# app/optimization/caching.py
import numpy as np

class SemanticCache:
    def __init__(self, threshold=0.92):
        self.threshold = threshold
        self.questions = []
        self.vectors = []
        self.answers = []
    
    def get(self, question: str) -> str | None:
        if not self.vectors:
            return None
        
        q_vec = embed_text(question)
        sims = np.dot(self.vectors, q_vec)
        best_idx = np.argmax(sims)
        
        if sims[best_idx] >= self.threshold:
            return self.answers[best_idx]
        return None
    
    def set(self, question: str, answer: str):
        self.questions.append(question)
        self.vectors.append(embed_text(question))
        self.answers.append(answer)
```

#### 3. Guardrails
```python
# app/guardrails/checks.py
import re

INJECTION_PATTERNS = [
    r"ignore (all )?previous (instructions|commands)",
    r"disregard (all )?previous",
    r"disregard your instructions",
]

def check_injection(text: str) -> bool:
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def check_output(answer: str, min_length: int = 10) -> str:
    if len(answer) < min_length:
        return "Câu trả lời quá ngắn."
    return answer
```

---

## 🚀 Sau 14 Ngày

### Bạn đã học được

1. **API Development**: FastAPI, endpoints, validation
2. **LLM Integration**: OpenAI API, streaming, structured output
3. **Local Serving**: Ollama, vLLM, OpenAI-compatible API
4. **RAG**: Embeddings, vector search, document retrieval
5. **Agentic AI**: LangGraph, query decomposition, grading
6. **Production**: Caching, routing, evaluation, guardrails

### Mở rộng tiếp

- [ ] Fine-tuning với QLoRA
- [ ] Gradio UI
- [ ] Deployment lên HF Spaces
- [ ] Multi-turn conversation memory
- [ ] Human-in-the-loop approval

---

## 📚 Tài Liệu Tham Khảo

- [OpenAI API Docs](https://platform.openai.com/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/)
- [Ollama](https://github.com/ollama/ollama)
- [FastAPI](https://fastapi.tiangolo.com/tutorial/)

---

## ⚠️ Common Pitfalls

1. **Embedding vs Completion keys**: Embedding luôn cần OpenAI key
2. **Qdrant :memory:**: Chỉ work trong 1 process
3. **Streaming**: Phải dùng `StreamingResponse` trong FastAPI
4. **Context length**: Giới hạn bởi model, 4o-mini = 128K tokens
