# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Cao Diệu Ly

**Nhóm:** 21-E403

**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có cosine similarity cao nghĩa là chúng gần như cùng hướng trong không gian vector, tức là nội dung ngữ nghĩa của hai câu rất giống nhau, dù cách diễn đạt có thể khác.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi thích học lập trình AI."
- Sentence B: "Trí tuệ nhân tạo là đam mê của tôi."
- Tại sao tương đồng: Khác nhau về từ vựng (AI/Trí tuệ nhân tạo) nhưng mang cùng một ý nghĩa sở thích.

**Ví dụ LOW similarity:**
- Sentence A: "Hôm nay trời rất đẹp."
- Sentence B: "Tôi đang code Python."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn độc lập (thời tiết vs lập trình).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì cosine similarity đo lường góc (hướng ngữ nghĩa), do đó không bị ảnh hưởng bởi độ dài (magnitude) của vector tài liệu như Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Stride (bước nhảy) = 500 - 50 = 450.
> Lượng ký tự cần nhảy sau chunk đầu tiên: 10,000 - 500 = 9,500.
> Số bước: ceil(9,500 / 450) = 22.
> Tổng số chunk: 1 (đầu) + 22 = 23 chunks.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Stride giảm còn 400, số lượng bước tăng lên, tổng chunk count sẽ là 25 chunks (tăng thêm). Chúng ta muốn overlap nhiều hơn để đảm bảo không cắt ngang đột ngột câu đang đọc lẻ tẻ, tránh việc agent thiếu đi bối cảnh đứng liền kề.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Truyện cổ tích Việt Nam

**Tại sao nhóm chọn domain này?**
> Nhóm chọn truyện cổ tích Việt Nam vì đây là dữ liệu phong phú về mặt hội thoại, nhân vật thiện/ác và các bài học đạo đức, đồng thời có format cấu trúc dễ nhận biết (các đoạn văn). Điều này giúp kiểm tra khả năng bắt được ngữ cảnh đầy đủ của Embeddings và dễ so sánh sự khác nhau giữa các Chunking strategies.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Cây Khế | [Link](https://truyencotich.top/doc-truyen/truyen-co-tich-cay-khe) | 3757 | `doc_id: "01_cay_khe.txt"`, `title: "Truyện Cây Khế"`, `main_character: "Người em"`, `theme: "Lòng tham"`, `moral_lesson: "Ở hiền gặp lành"`, `category: "Folk Tale"` |
| 2 | Sọ Dừa | [Link](https://truyencotich.top/doc-truyen/so-dua) | 12341 | `doc_id: "02_so_dua.txt"`, `title: "Truyện Sọ Dừa"`, `main_character: "Sọ Dừa"`, `theme: "Nhân hậu"`, `moral_lesson: "Xem trọng phẩm chất bên trong"`, `category: "Folk Tale"` |
| 3 | Tấm Cám | [Link](https://truyencotich.top/doc-truyen/truyen-co-tich-tam-cam) | 15434 | `doc_id: "03_tam_cam.txt"`, `title: "Truyện Tấm Cám"`, `main_character: "Cô Tấm"`, `theme: "Thiện thắng Ác"`, `moral_lesson: "Người hiền gặp lành"`, `category: "Folk Tale"` |
| 4 | Thạch Sanh | [Link](https://truyencotich.top/doc-truyen/thach-sanh-ly-thong) | 16026 | `doc_id: "04_thach_sanh_ly_thong.txt"`, `title: "Thạch Sanh"`, `main_character: "Thạch Sanh"`, `theme: "Dũng cảm"`, `moral_lesson: "Công lý và sự chính trực"`, `category: "Folk Tale"` |
| 5 | Thánh Gióng | [Link](https://truyencotich.top/doc-truyen/thanh-giong) | 10915 | `doc_id: "05_thanh_giong.txt"`, `title: "Thánh Gióng"`, `main_character: "Thánh Gióng"`, `theme: "Anh hùng"`, `moral_lesson: "Lòng yêu quốc và bảo vệ tổ quốc"`, `category: "Folk Tale"` |
| 6 | Cây tre trăm đốt | [Link](https://truyencotich.top/doc-truyen/cay-tre-tram-dot) | 10920 | `doc_id: "06_cay_tre_tram_dot.txt"`, `title: "Cây tre trăm đốt"`, `main_character: "Khoai"`, `theme: "Công lý"`, `moral_lesson: "Lên án sự gian tham, lừa lọc"`, `category: "Folk Tale"` |
| 7 | Sự tích trầu cau | [Link](https://truyencotich.top/doc-truyen/su-tich-trau-cau) | 5321 | `doc_id: "07_su_tich_trau_cau.txt"`, `title: "Sự tích trầu cau"`, `main_character: "Anh em họ Cao"`, `theme: "Tình thân"`, `moral_lesson: "Tình nghĩa anh em, vợ chồng ke sơn"`, `category: "Folk Tale"` |
| 8 | Em bé thông minh | [Link](https://truyencotich.top/doc-truyen/em-be-thong-minh) | 8742 | `doc_id: "08_em_be_thong_minh.txt"`, `title: "Em bé thông minh"`, `main_character: "Em bé"`, `theme: "Trí tuệ"`, `moral_lesson: "Khẳng định tài năng, trí tuệ dân gian"`, `category: "Folk Tale"` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | `str` | `"01_cay_khe.txt"` | Định danh duy nhất để quản lý vòng đời tài liệu (xóa/cập nhật). |
| `title` | `str` | `"Truyện Cây Khế"` | Cung cấp tiêu đề thân thiện để người dùng nhận diện nguồn thông tin. |
| `main_character`| `str` | `"Người em"` | Hỗ trợ lọc (filtering) các đoạn trích xoay quanh một nhân vật cụ thể. |
| `theme` | `str` | `"Lòng tham"` | Cho phép tìm kiếm hoặc phân tích theo chủ đề/nhân tâm của truyện. |
| `moral_lesson` | `str` | `"Ở hiền gặp lành"` | Hỗ trợ truy xuất các đoạn chứa nội dung giáo dục hoặc kết luận. |
| `category` | `str` | `"Folk Tale"` | Phân loại dữ liệu khi mở rộng hệ thống sang các thể loại khác (ngụ ngôn, truyền thuyết). |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy test trên 2 tài liệu minh chứng (`01_cay_khe.txt` và `04_thach_sanh_ly_thong.txt`):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `01_cay_khe.txt` | FixedSizeChunker (`fixed_size`) | 7 | 410.4 | Kém (Cắt ngang thân câu) |
| `01_cay_khe.txt` | SentenceChunker (`by_sentences`) | 11 | 260.2 | Có (Giữ được các câu đơn) |
| `04_thach_sanh.txt`| FixedSizeChunker (`fixed_size`) | 28 | 437.0 | Kém (Cắt ngang thân câu) |
| `04_thach_sanh.txt`| SentenceChunker (`by_sentences`) | 40 | 305.3 | Có (Top-1 baseline) |

### Strategy Của Tôi

**Loại:** Custom strategy (`ParagraphChunker`)

**Mô tả cách hoạt động:**
> `ParagraphChunker` hoạt động bằng cách ưu tiên chặt văn bản ở cấp độ đoạn văn thực thụ bằng regex chia đôi `/n/n` hoặc `/n`. Khối thuật toán đảm bảo giữ cụm văn bản lớn trừ khi đoạn văn vượt mức `max_chunk_size` mới bị chẻ ra thành các câu riêng lẻ.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn bản truyện cổ tích là một mạch truyện logic mà mỗi lời thoại, trạng thái diễn biến hành động đều gói gọn chung trong 1 đoạn văn lùi đầu dòng. Chiến lược này bảo toàn cấu trúc trọn vẹn ngữ nghĩa của cảnh huống, tránh gãy mạch hội thoại.

**Code snippet (nếu custom):**
```python
class ParagraphChunker(ChunkingStrategy):
    def __init__(self, max_chunk_size: int = 1500):
        self.max_chunk_size = max_chunk_size
        self._name = "paragraph"
        
    def chunk(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\n', text) if p.strip()]
        chunks = []
        for p in paragraphs:
            if len(p) > self.max_chunk_size:
                sub_chunks = [s.strip() + "." for s in p.split('. ') if s.strip()]
                chunks.extend(sub_chunks)
            else:
                chunks.append(p)
        return chunks
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `01_cay_khe.txt` | best baseline (Sentence) | 11 | 260.2 | Trung bình khá. Nhận được các cấu trúc đối thoại nhưng đôi lúc bị cắt đứt đại từ ("người đó", "nó"). |
| `01_cay_khe.txt` | **của tôi (Paragraph)** | 19 | 150.3 | Giữ cực kỳ vững vàng văn cảnh. Chunk gọn, mang lại context hoàn chỉnh. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Ly)  | ParagraphChunker | 9/10 | Giữ trọn mạch văn hóa truyện, trả đủ hội thoại mà tác giả muốn mô tả chung một ngữ cảnh. | Chunk length không đều. |
| Tuấn | SentenceChunker | 8.5/10 | Lọc chi tiết thông tin theo câu mượt mà | Dễ gây thiếu nội dung ở các cặp câu hội thoại mở-đóng hỏi đáp. |
| Nam | RecursiveChunker | 9.0/10 | Tự động resize tốt với separators | Có thể cắt ngang câu chuyện khi vượt ngưỡng size cứng. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> `ParagraphChunker` hoặc `RecursiveChunker` (khi tối ưu cấu hình `\n\n`) là hai lựa chọn tốt nhất. Vì chúng tôn trọng cấu trúc ngắt ý qua các đoạn lùi đầu dòng, phù hợp với truyện cổ tích có lượng mô tả / kịch bản đối thoại nằm liền mạch theo dòng hội thoại nhân vật.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex `r'(?<=[\.\!\?])(?:\s|\n)'` (Lookbehind) để chia tách văn bản tại vị trí ngay sau dấu kết thúc câu (`.`, `!`, `?`) kèm theo khoảng trắng hoặc dấu xuống dòng. Sau đó gom các nhóm câu lại theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Dựng hàm `_split` đệ quy: đi tuần tự từ level delimiter cao nhất (đoạn) đến cấp thấp (câu, chữ). Bất cứ lúc nào `len(text) < chunk_size`, thuật toán dừng và nối chunk, giải quyết vấn đề boundary cắt ngang.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ linh hoạt, hỗ trợ 2 lớp: in-memory (`list`) và `ChromaDB`. Nếu `chromadb` pip array được install, backend tự mount persistent client vào `./chroma_data`. Hàm search tính distance giữa vector input do `OpenAIEmbeddings()` nhả về với toàn bộ records của collection, rồi sắp xếp trả về `Top-K`.

**`search_with_filter` + `delete_document`** — approach:
> Dựa vào where filter object (chứa Key-Value dict của metadata JSON), loại bỏ những file không chứa đúng target metadata trước, sau đó mới đẩy vào tính cosin-similarly để giảm effort so khớp dư thừa và tăng tính Grounding triệt để. Xóa theo `doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:
> RAG implementation: format prompt `<context>{chunk_content}</context>`. Ép LLM model phải tuân thủ nghiêm ngặt instruction "Nếu không tìm thấy, trả lời 'Không có thông tin'" thay vì bịa câu trả lời. 

### Test Results

```
platform win32 -- Python 3.13.2, pytest-9.0.2, pluggy-1.6.0
==================== 42 passed in 1.15s ====================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

Kết quả thực nghiệm bằng cách sử dụng module `OpenAIEmbedder (text-embedding-3-small)` từ script chạy tự động.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Ăn một quả, trả cục vàng. | May túi ba gang, mang đi mà đựng. | high | 0.4626 | Yes |
| 2 | Bống bống bang bang. | Lên ăn cơm vàng cơm bạc nhà ta. | high | 0.3918 | Yes |
| 3 | Khắc nhập, khắc nhập! | Một trăm đốt tre đã dính liền với nhau. | high | 0.2626 | Partial |
| 4 | Thạch Sanh bắn đại bàng. | Người nông phu cặm cụi làm việc. | low | 0.3036 | Yes |
| 5 | Thánh Gióng về trời. | Học trò xâu chỉ qua vỏ ốc. | low | 0.2992 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Bất ngờ nhất là Pair 3 tuy có liên quan mật thiết về truyện (Cây tre trăm đốt) nhưng độ tương đồng chỉ ở mức 0.26, thậm chí thấp hơn Pair 4 (khi kết hợp ngẫu nhiên). Điều này chỉ ra rằng Embedding dựa vào ngữ nghĩa của TỪ vựng hình thành câu chứ không hiểu "kiến thức tác phẩm" nếu từ vựng quá xa vời (câu thần chú ngữ âm "khắc nhập" vs câu miêu tả "đốt tre dính liền"). 

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân (`ParagraphChunker` + `OpenAI Embedding`), CSDL ChromaDB, KHÔNG áp dụng Filter Metadata để đánh giá độ gắt.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Nhân vật chính trong truyện Tấm Cám là ai? | Tấm và Cám |
| 2 | Thánh Gióng chống giặc như thế nào? | Cưỡi ngựa sắt, nhổ tre đánh giặc |
| 3 | Bài học từ truyện Sọ Dừa? | Đề cao vẻ đẹp phẩm chất bên trong |
| 4 | Cây Khế dạy gì về lòng biết ơn? | Phải biết đền ơn đáp nghĩa người giúp mình |
| 5 | Em bé thông minh giải đố ra sao? | Bằng trí thông minh dân gian sắc sảo |

### Kết Quả Của Tôi (Raw Execution no Filter)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Nhân vật chính truyện Tấm Cám?| "Mẹ Tấm mất sớm, sau đó mấy năm cha Tấm... ở với dì ghẻ là mẹ Cám." | 0.6179 | Yes | Nhân vật là Tấm (và mẹ Cám). |
| 2 | Thánh Gióng chống giặc? | "Khi đến nơi này, Gióng đem áo giáp sắt cùng nón sắt để lại... bay thẳng về trời." | 0.5952 | Partial | Đã trả mã giáp, bay về trời sau khi đánh giặc. |
| 3 | Bài học Sọ Dừa? | "Đến tối, khi nến thắp sáng... Sọ Dừa từ phòng bước ra là chàng thanh niên..." | 0.6396 | No | Chàng Sọ Dừa là thanh niên (sai trọng tâm bài học). |
| 4 | Cây Khế lòng biết ơn? | "Để đền ơn Thạch Sanh, Thủy vương... biếu đàn thần." | 0.5065 | No | Thủy vương trả ơn Thạch Sanh (Lệch truyện!). |
| 5 | Em bé thông minh giải đố? | "Quả nhiên, Thạch Sanh bị bắt, giải về trình Lý Thông." | 0.4751 | No | Thạch Sanh bị Lý Thông bắt (Lệch truyện!). |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5
(Giải thích: Các câu hỏi 3, 4, 5 truy xuất ra các chunk có các chữ "thông minh", "đền ơn", nhưng ở các truyện khác do sự chồng chéo ngữ nghĩa, minh chứng rõ ràng RAG nếu thiếu Metadata Filter sẽ cực kỳ tệ hại ở kho data hỗn hợp).

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được rằng việc cắt Chunking dù có đẹp đến đâu (như ParagraphChunker), nhưng nếu tập dữ liệu RAG có nhiều tài liệu chung một thể loại (Folklore), Embedding Similarity thuần tuý sẽ rất dễ truy xuất các sự kiện đánh nhau, đền ơn từ một truyện khác gán cho truyện đang bị hỏi làm nhiễu Agent hoàn toàn.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Thực tiễn cho thấy bắt buộc phải sử dụng Hybrid Search (hỗ trợ Exact Keyword mapping) hoặc áp dụng Filter Query via Metadata (bắt Agent truyền vào `{ "doc_id": "01_cay_khe.txt" }` trước khi tìm kím) thì truy xuất mới chính xác được 5/5. Đây là điểm mạnh tuyệt đối của ChromaDB (sử dụng parameter `where`).

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ cập nhật Agent Flow: Trước khi Retrieval, tôi sẽ gọi LLM 1 lần để nhổ (extract) filter metadata từ câu hỏi (Ví dụ: "Truyện Sọ Dừa" -> Extract `{"title": "Truyện Sọ Dừa"}`). Chuyền filter này vào hàm search thì kết quả sẽ relevant tuyệt đối thay vì dính điểm sai như hiện tại.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **90 / 100** |
