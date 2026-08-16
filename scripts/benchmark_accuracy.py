import numpy as np
from ggmbed import GGMBedEncoder, DenseEncoder

SENTENCES = [
    "Dense embeddings represent semantic meaning in a vector space.",
    "llama.cpp is a highly optimized library written in C++ for CPU inference.",
    "Semantic search systems retrieve information based on context rather than keyword matching.",
    "Zero-PyTorch inference reduces deployment memory footprint significantly.",
    "cosine similarity is defined as the dot product of two normalized vectors.",
    "Different quantization types offer different trade-offs between speed, size, and precision.",
    "Pre-built binary wheels are optimized for AVX2 and NEON instructions.",
    "This library is a drop-in replacement for fastembed.",
    "CLS pooling extracts the first token representation as the sentence embedding.",
    "Mean pooling averages all non-padded token vectors."
]

def main():
    print("Loading F32 baseline model...")
    enc_f32 = DenseEncoder("sentence-transformers/all-MiniLM-L6-v2", quantization="F32")
    embs_f32 = enc_f32.encode(SENTENCES, normalize=True)

    quant_types = ["F16", "Q8_0", "Q4_0"]
    
    print("\n--- Accuracy Comparison vs F32 (Mean Cosine Similarity) ---")
    print(f"{'Quantization':<15} | {'Cosine Sim vs F32':<20} | {'Status':<15}")
    print("-" * 56)
    
    for quant in quant_types:
        try:
            enc_q = DenseEncoder("sentence-transformers/all-MiniLM-L6-v2", quantization=quant)
            embs_q = enc_q.encode(SENTENCES, normalize=True)
            
            # Compute cosine similarities for each sentence
            sims = []
            for i in range(len(SENTENCES)):
                v1 = embs_f32[i]
                v2 = embs_q[i]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                sims.append(sim)
            
            mean_sim = np.mean(sims)
            
            # Determine loss status
            if mean_sim >= 0.999:
                status = "Near Lossless"
            elif mean_sim >= 0.995:
                status = "Excellent"
            elif mean_sim >= 0.98:
                status = "Good"
            else:
                status = "Not Recommended"
                
            print(f"{quant:<15} | {mean_sim:18.6f} | {status:<15}")
        except Exception as e:
            print(f"{quant:<15} | Failed: {e}")

if __name__ == "__main__":
    main()
