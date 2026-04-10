import os
from dotenv import load_dotenv
from src.models import Document
from src.chunking import ParagraphChunker
from src.embeddings import OpenAIEmbedder
from src.store import EmbeddingStore

def run_rag_benchmark():
    # Thiết lập encoding cho Windows
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("--- LOI: Chua tim thay OPENAI_API_KEY trong file .env ---")
        return

    # 1. Khởi tạo
    embedder = OpenAIEmbedder()
    # Sử dụng collection name tạm thời để tránh ảnh hưởng dữ liệu cũ
    store = EmbeddingStore(collection_name="benchmark_experiment", embedding_fn=embedder)
    chunker = ParagraphChunker(max_chunk_size=1500)
    
    # 2. Danh sách dữ liệu (8 truyện)
    data_files = [
        "01_cay_khe.txt", "02_so_dua.txt", "03_tam_cam.txt", 
        "04_thach_sanh_ly_thong.txt", "05_thanh_giong.txt",
        "06_cay_tre_tram_dot.txt", "07_su_tich_trau_cau.txt",
        "08_em_be_thong_minh.txt"
    ]
    
    print("--- Dang nap va index 8 tai lieu truyen co tich ---")
    for filename in data_files:
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            print(f"--- Thieu file: {filename} ---")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            chunks = chunker.chunk(content)
            docs = [Document(id=f"{filename}_{i}", content=c, metadata={"doc_id": filename}) 
                    for i, c in enumerate(chunks)]
            store.add_documents(docs)

    # 3. Benchmark Queries
    queries = [
        "Nhân vật chính trong truyện Tấm Cám là ai?",
        "Thánh Gióng chống giặc như thế nào?",
        "Bài học từ truyện Sọ Dừa?",
        "Cây Khế dạy gì về lòng biết ơn?",
        "Em bé thông minh giải đố ra sao?"
    ]

    print("\n--- KẾT QUẢ TRUY XUẤT (RAW EXECUTION - NO FILTER) ---")
    print("-" * 100)
    print(f"{'#':<3} | {'Query':<40} | {'Score':<8} | {'Source':<20}")
    print("-" * 100)
    
    for i, q in enumerate(queries, 1):
        results = store.search(q, top_k=1)
        if results:
            r = results[0]
            source = r['metadata'].get('doc_id', 'N/A')
            print(f"{i:<3} | {q:<40} | {r['score']:.4f} | {source:<20}")
            print(f"   [Noi dung truy xuat]: {r['content'][:300]}...")
            print("-" * 50)
        else:
            print(f"{i:<3} | {q:<40} | {'N/A':<8} | {'None':<20}")

if __name__ == "__main__":
    print("=== THUC NGHIEM CHAT LUONG TRUY XUAT (RAG BENCHMARK) ===")
    run_rag_benchmark()
