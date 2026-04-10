from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    Một agent trả lời các câu hỏi bằng cách sử dụng cơ sở tri thức vector.

    Mô hình tạo nội dung tăng cường truy xuất (RAG):
        1. Truy xuất top-k đoạn văn bản liên quan từ kho lưu trữ.
        2. Xây dựng prompt với các đoạn văn bản làm ngữ cảnh.
        3. Gọi LLM để tạo câu trả lời.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # Bước 1: Truy xuất top-k đoạn văn bản liên quan từ kho lưu trữ
        results = self._store.search(question, top_k=top_k)

        # Bước 2: Xây dựng prompt với ngữ cảnh đã truy xuất
        context_parts = []
        for i, result in enumerate(results, start=1):
            context_parts.append(f"[{i}] {result['content']}")

        context = "\n\n".join(context_parts)

        prompt = (
            "Sử dụng ngữ cảnh sau đây để trả lời câu hỏi. "
            "Nếu ngữ cảnh không chứa đủ thông tin, hãy nêu rõ điều đó.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Câu trả lời:"
        )

        # Bước 3: Gọi LLM để tạo câu trả lời
        return self._llm_fn(prompt)
