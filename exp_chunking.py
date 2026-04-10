import os
from src.chunking import ChunkingStrategyComparator

def run_chunking_experiment():
    # Thiết lập encoding cho Windows
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    comparator = ChunkingStrategyComparator()
    
    files = [
        ("01_cay_khe.txt", "Cây Khế"),
        ("04_thach_sanh_ly_thong.txt", "Thạch Sanh")
    ]
    
    # Kích thước chunk dùng trong so sánh baseline (thường là 200 hoặc 500 ký tự tùy ngữ cảnh)
    # Trong REPORT, các số liệu FixedSize thường là quanh mức 500 ký tự để đạt 23-28 chunks.
    CHUNK_SIZE = 500 

    for filename, title in files:
        path = os.path.join("data", filename)
        if not os.path.exists(path):
            print(f"⚠️ Không tìm thấy file: {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            
        print(f"\n📊 KẾT QUẢ CHO TÀI LIỆU: {title} ({filename})")
        print("-" * 60)
        results = comparator.compare(text, chunk_size=CHUNK_SIZE)
        
        print(f"{'Strategy':<20} | {'Chunk Count':<12} | {'Avg Length':<12}")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:<20} | {metrics['count']:<12} | {metrics['avg_length']:<12}")
            
        # Hiển thị nội dung thực tế của Paragraph strategy để kiểm chứng ngữ cảnh
        print(f"\n📝 NOI DUNG 3 CHUNK DAU TIEN CUA ParagraphChunker:")
        p_chunks = results['paragraph']['chunks']
        for idx, chunk in enumerate(p_chunks[:3], 1):
            print(f"--- Chunk {idx} ---")
            print(chunk)
            print("-" * 20)

if __name__ == "__main__":
    print("=== THUC NGHIEM SO SANH CHIEN LUOC CHUNKING (BASELINE ANALYSIS) ===")
    run_chunking_experiment()
