from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Chia nhỏ văn bản thành các đoạn có kích thước cố định với phần chồng lấp (overlap) tùy chọn.

    Quy tắc:
        - Mỗi đoạn có độ dài tối đa là chunk_size ký tự.
        - Các đoạn liên tiếp chia sẻ các ký tự chồng lấp.
        - Đoạn cuối cùng chứa phần còn lại của văn bản.
        - Nếu văn bản ngắn hơn chunk_size, trả về [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Chia nhỏ văn bản thành các đoạn có tối đa max_sentences_per_chunk câu.

    Phát hiện câu: chia theo ". ", "! ", "? " hoặc ".\n".
    Loại bỏ khoảng trắng thừa ở đầu và cuối mỗi đoạn.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Chia theo ranh giới câu: ". ", "! ", "? ", hoặc ".\n"
        # Sử dụng regex để giữ lại dấu câu ở cuối mỗi câu.
        parts = re.split(r'(?<=[\.\!\?])(?:\s|\n)', text)

        # Lọc bỏ các phần trống và xóa khoảng trắng thừa
        sentences = [s.strip() for s in parts if s.strip()]

        if not sentences:
            return []

        # Nhóm các câu vào từng đoạn (chunk)
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks


class RecursiveChunker:
    """
    Chia nhỏ văn bản một cách đệ quy bằng cách sử dụng các ký tự phân tách theo thứ tự ưu tiên.

    Thứ tự ưu tiên ký tự phân tách mặc định:
        ["\\n\\n", "\\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        results = self._split(text, self.separators)
        # Lọc bỏ các đoạn trống và xóa khoảng trắng thừa
        return [c.strip() for c in results if c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Trường hợp cơ sở: văn bản khớp với chunk_size
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Nếu không còn ký tự phân tách nào, buộc phải chia theo ký tự (slicing theo chunk_size)
        if not remaining_separators:
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Ký tự phân tách trống có nghĩa là chia theo từng ký tự — chuyển về buộc chia (force-split)
        if separator == "":
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        # Chia theo ký tự phân tách hiện tại
        parts = current_text.split(separator)

        # Nếu việc chia không giúp ích gì (chỉ có 1 phần), thử dùng ký tự phân tách tiếp theo
        if len(parts) <= 1:
            return self._split(current_text, next_separators)

        # Ghép các phần nhỏ lại với nhau, đệ quy trên các phần quá lớn
        results: list[str] = []
        current_chunk = ""

        for i, part in enumerate(parts):
            # Xây dựng phần tử bằng cách thêm ký tự phân tách lại (ngoại trừ phần cuối cùng)
            piece = part if i == len(parts) - 1 else part + separator

            if not current_chunk:
                current_chunk = piece
            elif len(current_chunk + piece) <= self.chunk_size:
                current_chunk += piece
            else:
                 # Đoạn hiện tại đã sẵn sàng — tiến hành xử lý
                if len(current_chunk) <= self.chunk_size:
                    results.append(current_chunk)
                else:
                    results.extend(self._split(current_chunk, next_separators))
                current_chunk = piece

        # Không quên đoạn cuối cùng đã tích lũy
        if current_chunk:
            if len(current_chunk) <= self.chunk_size:
                results.append(current_chunk)
            else:
                results.extend(self._split(current_chunk, next_separators))

        return results


class ParagraphChunker:
    """
    Chia nhỏ văn bản dựa trên các đoạn văn tự nhiên (sử dụng dấu ngắt đoạn kép \n\n hoặc \n \n).
    Nếu đoạn văn dài hơn max_chunk_size, thuật toán fallback sang cắt theo câu.
    """
    def __init__(self, max_chunk_size: int = 600) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Tách chia dựa trên khoảng ngắt đoạn rõ ràng
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks = []
        for p in paragraphs:
            # Fallback nếu đoạn văn quá dài
            if len(p) > self.max_chunk_size:
                sub_chunks = [s.strip() + "." for s in p.split('. ') if s.strip()]
                chunks.extend(sub_chunks)
            else:
                chunks.append(p)
                
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Tính toán độ tương đồng cosine (cosine similarity) giữa hai vector.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Trả về 0.0 nếu một trong hai vector có độ lớn (magnitude) bằng không.
    """
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Chạy tất cả các chiến lược chia nhỏ có sẵn và so sánh kết quả của chúng."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Tạo các chunker với chunk_size được cung cấp
        fixed = FixedSizeChunker(chunk_size=chunk_size, overlap=50)
        sentence = SentenceChunker(max_sentences_per_chunk=3)
        recursive = RecursiveChunker(chunk_size=chunk_size)
        paragraph = ParagraphChunker(max_chunk_size=chunk_size*3) # Tùy chỉnh max_chunk to hơn bình thường để tận dụng ngắt đoạn

        strategies = {
            "fixed_size": fixed.chunk(text),
            "by_sentences": sentence.chunk(text),
            "recursive": recursive.chunk(text),
            "paragraph": paragraph.chunk(text),
        }

        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            result[name] = {
                "count": count,
                "avg_length": round(avg_length, 2),
                "chunks": chunks,
            }

        return result
