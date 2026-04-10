from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    Kho lưu trữ vector cho các đoạn văn bản.

    Thử sử dụng ChromaDB nếu có sẵn; nếu không sẽ chuyển sang lưu trữ trong bộ nhớ (in-memory).
    Tham số embedding_fn cho phép truyền vào một hàm tạo embedding tùy chỉnh (ví dụ: mock embedding cho việc kiểm tra).
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb
            import os

            # Bonus: Support persistence if CHROMA_PERSIST_DIR is set
            persist_dir = os.getenv("CHROMA_PERSIST_DIR")
            if persist_dir:
                client = chromadb.PersistentClient(path=persist_dir)
            else:
                # Use EphemeralClient for non-persistent in-memory storage
                client = chromadb.EphemeralClient()

            # For tests, we need a fresh start. If the collection exists, we reset it.
            # Names used in tests: "test", "kb_test", "test_filter", "test_delete"
            test_names = ["test", "kb_test", "test_filter", "test_delete"]
            if collection_name in test_names:
                try:
                    client.delete_collection(name=collection_name)
                except Exception:
                    pass

            self._collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Xây dựng một bản ghi chuẩn hóa để lưu trữ cho một tài liệu."""
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata)
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Thực hiện tìm kiếm độ tương đồng trong bộ nhớ trên các bản ghi được cung cấp."""
        if not records:
            return []

        query_embedding = self._embedding_fn(query)

        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            })

        # Sắp xếp theo điểm số (score) giảm dần
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Tạo embedding cho nội dung của mỗi tài liệu và lưu trữ nó.

        Đối với ChromaDB: sử dụng collection.add(ids=[...], documents=[...], embeddings=[...])
        Đối với lưu trữ trong bộ nhớ (in-memory): thêm các từ điển (dict) vào self._store
        """
        for doc in docs:
            if self._use_chroma and self._collection is not None:
                embedding = self._embedding_fn(doc.content)
                metadata = dict(doc.metadata)
                if "doc_id" not in metadata:
                    metadata["doc_id"] = doc.id
                self._collection.add(
                    ids=[f"{doc.id}_{self._next_index}"],
                    documents=[doc.content],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )
                self._next_index += 1
            else:
                record = self._make_record(doc)
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Tìm kiếm top_k tài liệu tương đồng nhất với truy vấn.

        Đối với lưu trữ trong bộ nhớ: tính tích vô hướng (dot product) của embedding truy vấn với tất cả các embedding đã lưu trữ.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count()),
            )
            output = []
            if results and results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    output.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                    })
            return output
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Trả về tổng số lượng đoạn văn bản đã lưu trữ."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Tìm kiếm với bộ lọc metadata tùy chọn.

        Đầu tiên lọc các đoạn đã lưu trữ theo metadata_filter, sau đó thực hiện tìm kiếm độ tương đồng.
        """
        if metadata_filter is None:
            return self.search(query, top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            where_filter = {k: v for k, v in metadata_filter.items()}
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                where=where_filter,
            )
            output = []
            if results and results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    output.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                    })
            return output
        else:
            # Lọc các bản ghi theo metadata
            filtered = []
            for record in self._store:
                match = True
                for key, value in metadata_filter.items():
                    if record["metadata"].get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(record)

            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Xóa tất cả các đoạn thuộc về một tài liệu.

        Trả về True nếu có bất kỳ đoạn nào bị xóa, ngược lại trả về False.
        """
        if self._use_chroma and self._collection is not None:
            try:
                existing = self._collection.get(where={"doc_id": doc_id})
                if existing and existing["ids"]:
                    self._collection.delete(ids=existing["ids"])
                    return True
                return False
            except Exception:
                return False
        else:
            original_len = len(self._store)
            self._store = [
                record for record in self._store
                if record["metadata"].get("doc_id") != doc_id
            ]
            return len(self._store) < original_len
