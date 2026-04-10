import os
from dotenv import load_dotenv
from src.embeddings import OpenAIEmbedder
from src.chunking import compute_similarity

def run_similarity_experiment():
    # Thiết lập encoding cho Windows
    import sys
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("--- LOI: Chua tim thay OPENAI_API_KEY trong file .env ---")
        return

    embedder = OpenAIEmbedder()
    
    pairs = [
        ("Ăn một quả, trả cục vàng.", "May túi ba gang, mang đi mà đựng."),
        ("Bống bống bang bang.", "Lên ăn cơm vàng cơm bạc nhà ta."),
        ("Khắc nhập, khắc nhập!", "Một trăm đốt tre đã dính liền với nhau."),
        ("Thạch Sanh bắn đại bàng.", "Người nông phu cặm cụi làm việc."),
        ("Thánh Gióng về trời.", "Học trò xâu chỉ qua vỏ ốc.")
    ]

    print(f"{'#':<3} | {'Sentence A':<35} | {'Sentence B':<45} | {'Score':<8}")
    print("-" * 100)
    
    for i, (a, b) in enumerate(pairs, 1):
        vec_a = embedder(a)
        vec_b = embedder(b)
        score = compute_similarity(vec_a, vec_b)
        print(f"{i:<3} | {a:<35} | {b:<45} | {score:.4f}")

if __name__ == "__main__":
    print("=== THUC NGHIEM DO TUONG DONG (SIMILARITY PREDICTIONS) ===")
    run_similarity_experiment()
