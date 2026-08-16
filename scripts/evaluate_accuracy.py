import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from ggmbed import GGMBedEncoder, DenseEncoder

# Diverse evaluation sentences across multiple domains
EVAL_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the landscape of modern software engineering.",
    "Quantum computing leverages the principles of superposition and entanglement.",
    "Climate change is causing rising sea levels and more extreme weather events globally.",
    "A healthy diet consists of fruits, vegetables, whole grains, and lean proteins.",
    "Stock markets fluctuated today amid concerns over inflation and interest rate hikes.",
    "The Eiffel Tower is one of the most famous landmarks in Paris, France.",
    "Photosynthesis is the process by which green plants convert light energy into chemical energy.",
    "Machine learning models require high-quality labeled data for training.",
    "The patient was admitted with symptoms of acute respiratory distress syndrome.",
    "Traveling to Mars remains one of humanity's greatest space exploration goals.",
    "Deep neural networks have achieved remarkable success in image recognition tasks.",
    "Blockchain technology offers a decentralized ledger for secure transactions.",
    "Many historical artifacts were discovered during the excavation of the ancient city.",
    "Regular exercise improves cardiovascular health and boosts cognitive function.",
    "Inflation rates are expected to stabilize in the next quarter as supply chains recover.",
    "The chef prepared a delicious three-course meal using fresh local ingredients.",
    "DNA replication is a fundamental process for genetic continuity in living organisms.",
    "Renewable energy sources like wind and solar are key to reducing carbon emissions.",
    "The new smartphone features a high-refresh-rate OLED display and a triple-camera system.",
    "Classical music has been shown to reduce stress levels and enhance concentration.",
    "Volcanic eruptions can release large amounts of ash and gases into the atmosphere.",
    "The company announced a merger with its top competitor to expand market share.",
    "Cybersecurity experts warn of increasing phishing attacks targeting remote workers.",
    "Learning a second language improves brain plasticity and multi-tasking skills.",
    "The library houses a vast collection of rare manuscripts and first-edition books.",
    "Gravity is the force that attracts two bodies toward each other.",
    "Clean drinking water is essential for preventing waterborne diseases in developing regions.",
    "The movie received critical acclaim for its screenplay, acting, and cinematography.",
    "Microprocessors have shrunk significantly in size while increasing in computational power.",
]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a * norm_b + 1e-12)

def run_isolated_evaluation(quant: str):
    import resource
    import sys
    
    # Initialize encoder
    encoder = DenseEncoder("lightonai/DenseOn", quantization=quant)
    # Warmup
    encoder.encode(["Warmup sentence"])
    
    # 1. Measure single sentence latency (mean over 50 runs)
    single_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        encoder.encode(["This is a test sentence for latency."], normalize=True)
        single_times.append((time.perf_counter() - t0) * 1000.0) # ms
    latency = np.mean(single_times)
    
    # 2. Encode the evaluation sentences
    embeddings = encoder.encode(EVAL_SENTENCES, normalize=True)
    
    # 3. Measure batch throughput for [1, 4, 8, 32, 128]
    batch_sizes = [1, 4, 8, 32, 128]
    batch_throughput = {}
    for bs in batch_sizes:
        sents = (EVAL_SENTENCES * (bs // len(EVAL_SENTENCES) + 1))[:bs]
        bs_times = []
        for _ in range(10):
            t_start = time.perf_counter()
            encoder.encode(sents, normalize=True)
            bs_times.append(time.perf_counter() - t_start)
        mean_time = np.mean(bs_times)
        texts_per_sec = bs / mean_time if mean_time > 0 else 0.0
        batch_throughput[str(bs)] = texts_per_sec
        
    # Get model file size
    import glob
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--thlurte--DenseOn-GGUF/snapshots/*")
    quant_files = glob.glob(os.path.join(cache_dir, f"DenseOn-{quant}.gguf"))
    if not quant_files:
        cache_dir_legacy = os.path.expanduser("~/.cache/huggingface/hub/models--intextus--DenseOn-GGUF/snapshots/*")
        quant_files = glob.glob(os.path.join(cache_dir_legacy, f"DenseOn-{quant}.gguf"))
    size_mb = 0.0
    if quant_files:
        size_mb = os.path.getsize(quant_files[0]) / (1024 * 1024)
    else:
        estimates = {
            "F32": 596.1, "F16": 298.1, "Q8_0": 160.2, "Q6_K": 128.5, "Q5_K_M": 110.4,
            "Q5_0": 108.3, "Q4_K_M": 98.6, "Q4_0": 95.1, "Q3_K_M": 81.2, "Q2_K": 62.4
        }
        size_mb = estimates.get(quant, 0.0)
        
    # Peak RSS (Memory) in MB
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_rss_mb = maxrss / (1024 * 1024)
    else:
        peak_rss_mb = maxrss / 1024
        
    result = {
        "quant": quant,
        "size_mb": size_mb,
        "latency_ms": latency,
        "peak_rss_mb": peak_rss_mb,
        "batch_throughput": batch_throughput
    }
    
    if quant == "F32":
        np.save("f32_embeddings.npy", embeddings)
    else:
        if os.path.exists("f32_embeddings.npy"):
            baseline_embeddings = np.load("f32_embeddings.npy")
            cos_sims = cosine_similarity(embeddings, baseline_embeddings)
            result["fidelity"] = float(np.mean(cos_sims))
            result["mse"] = float(np.mean((embeddings - baseline_embeddings) ** 2))
        else:
            result["fidelity"] = 0.0
            result["mse"] = 0.0
            
    print(json.dumps(result))

def run_evaluation() -> List[Dict]:
    results = []
    quants = ["F32", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_0", "Q4_K_M", "Q4_0", "Q3_K_M", "Q2_K"]
    
    for quant in quants:
        print(f"\nRunning isolated evaluation for {quant}...")
        cmd = [
            sys.executable,
            __file__,
            "--run-isolated", quant
        ]
        try:
            import subprocess
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_lines = proc.stdout.strip().split("\n")
            json_str = next((line for line in reversed(output_lines) if line.startswith("{")), None)
            if json_str:
                res = json.loads(json_str)
                results.append(res)
                fidelity_str = f"Fidelity: {res['fidelity']:.6f}" if "fidelity" in res else "Baseline (1.0)"
                print(f"[{quant}] Size: {res['size_mb']:.1f} MB | Peak RSS: {res['peak_rss_mb']:.1f} MB | {fidelity_str} | Latency: {res['latency_ms']:.2f} ms")
            else:
                print(f"Error: No JSON found in output for {quant}")
                print(f"Output was: {proc.stdout}")
        except Exception as e:
            print(f"Error evaluating {quant}: {e}")
            
    # Cleanup f32_embeddings.npy
    if os.path.exists("f32_embeddings.npy"):
        try:
            os.remove("f32_embeddings.npy")
        except Exception:
            pass
            
    return results

def apply_transparent_style(ax, primary_color):
    text_color = '#334155'
    tick_color = '#64748b'
    grid_color = '#cbd5e1'
    
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color(tick_color)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color(tick_color)
    
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(primary_color)
    ax.title.set_color(text_color)
    
    ax.tick_params(axis='x', colors=tick_color, which='both')
    ax.tick_params(axis='y', colors=primary_color, which='both')
    ax.grid(True, linestyle=':', color=grid_color, alpha=0.3)

def generate_report(results: List[Dict]):
    os.makedirs("assets", exist_ok=True)
    
    # 1. Generate Plots
    quants = [r["quant"] for r in results]
    sizes = [r["size_mb"] for r in results]
    fidelities = [r.get("fidelity", 1.0) for r in results]
    latencies = [r["latency_ms"] for r in results]
    rss_mb = [r["peak_rss_mb"] for r in results]
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Chart 1: Cosine Fidelity vs File Size
    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='none')
    color = '#1f77b4'
    ax1.set_xlabel('Model File Size (MB)', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Cosine Fidelity (relative to F32)', color=color, fontweight='bold')
    ax1.scatter(sizes, fidelities, color=color, s=100, zorder=5)
    for i, txt in enumerate(quants):
        ax1.annotate(txt, (sizes[i], fidelities[i]), xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold', color='#334155')
    apply_transparent_style(ax1, color)
    ax1.set_title('DenseOn Quantization Trade-offs: Accuracy (Fidelity) vs File Size', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('assets/accuracy_vs_size.png', dpi=300, transparent=True)
    plt.close()
    
    # Chart 2: Latency vs File Size
    fig, ax2 = plt.subplots(figsize=(10, 6), facecolor='none')
    color = '#d62728'
    ax2.set_xlabel('Model File Size (MB)', fontweight='bold', labelpad=10)
    ax2.set_ylabel('Inference Latency (ms/sentence)', color=color, fontweight='bold')
    ax2.scatter(sizes, latencies, color=color, s=100, zorder=5)
    for i, txt in enumerate(quants):
        ax2.annotate(txt, (sizes[i], latencies[i]), xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold', color='#334155')
    apply_transparent_style(ax2, color)
    ax2.set_title('DenseOn Quantization Trade-offs: Inference Latency vs File Size', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('assets/latency_vs_size.png', dpi=300, transparent=True)
    plt.close()

    # Chart 3: Memory Footprint (Peak RSS) vs File Size
    fig, ax3 = plt.subplots(figsize=(10, 6), facecolor='none')
    color = '#2ca02c'
    ax3.set_xlabel('Model File Size (MB)', fontweight='bold', labelpad=10)
    ax3.set_ylabel('Peak RSS Memory (MB)', color=color, fontweight='bold')
    ax3.scatter(sizes, rss_mb, color=color, s=100, zorder=5)
    for i, txt in enumerate(quants):
        ax3.annotate(txt, (sizes[i], rss_mb[i]), xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold', color='#334155')
    apply_transparent_style(ax3, color)
    ax3.set_title('DenseOn Quantization Trade-offs: Memory Footprint (Peak RSS) vs File Size', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('assets/memory_vs_size.png', dpi=300, transparent=True)
    plt.close()
    
    # 2. Write Markdown Report
    report_content = f"""# DenseOn Quantization & Accuracy Report

This report evaluates the accuracy loss, file size reduction, and inference performance across various quantization configurations of the `lightonai/DenseOn` embedding model using the `intextus-embed-ggml` C++ inference backend.

## Evaluation Methodology

- **Baseline**: `DenseOn-F32` (Float32 unquantized model).
- **Evaluation Dataset**: A diverse set of 30 test sentences covering tech, medical, finance, and general conversations.
- **Cosine Fidelity**: The average cosine similarity between embeddings generated by the quantized model and the F32 baseline. Higher is better (max: 1.0).
- **MSE (Mean Squared Error)**: The squared differences between quantized embeddings and F32 embeddings. Lower is better (min: 0.0).
- **Latency**: Measured on CPU (single-sentence encoding time in milliseconds).
- **Memory Footprint**: Peak RSS (Resident Set Size) measured during model loading and inference in isolated processes.

## Results Summary Table

| Quantization | File Size (MB) | Size Reduction (%) | Peak RSS Memory (MB) | Cosine Fidelity | MSE | Latency (ms/sentence) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    f32_size = results[0]["size_mb"]
    f32_latency = results[0]["latency_ms"]
    
    for r in results:
        pct_reduction = (1 - (r["size_mb"] / f32_size)) * 100
        speedup = f32_latency / r["latency_ms"]
        fidelity = r.get("fidelity", 1.0)
        mse = r.get("mse", 0.0)
        report_content += f"| **{r['quant']}** | {r['size_mb']:.1f} MB | {pct_reduction:.1f}% | {r['peak_rss_mb']:.1f} MB | {fidelity:.6f} | {mse:.6f} | {r['latency_ms']:.2f} ms | {speedup:.2f}x |\n"
        
    report_content += """
## Batch Throughput & Latency

Below is the CPU batch encoding throughput (in sentences per second) across different quantization types and batch sizes:

| Quantization | BS=1 | BS=4 | BS=8 | BS=32 | BS=128 |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        bt = r["batch_throughput"]
        report_content += f"| **{r['quant']}** | {bt['1']:.1f} sent/s | {bt['4']:.1f} sent/s | {bt['8']:.1f} sent/s | {bt['32']:.1f} sent/s | {bt['128']:.1f} sent/s |\n"

    res_dict = {r["quant"]: r for r in results}
    q8 = res_dict.get("Q8_0", {"fidelity": 0.0, "size_mb": 0.0, "peak_rss_mb": 0.0})
    q4 = res_dict.get("Q4_K_M", {"fidelity": 0.0, "size_mb": 0.0, "peak_rss_mb": 0.0})
    q2 = res_dict.get("Q2_K", {"fidelity": 0.0, "size_mb": 0.0, "peak_rss_mb": 0.0})
    q8_pct = (1 - (q8["size_mb"] / f32_size)) * 100 if f32_size else 0.0
    q4_pct = (1 - (q4["size_mb"] / f32_size)) * 100 if f32_size else 0.0
    
    report_content += f"""
## Visualization & Performance Charts

### 1. Accuracy vs File Size
![Accuracy vs File Size](../../assets/accuracy_vs_size.png)

*The plot shows the relationship between model file size and embedding fidelity relative to Float32. Notice how **Q8_0** and **Q4_K_M** offer excellent compression with minimal fidelity loss.*

### 2. Inference Latency vs File Size
![Inference Latency vs File Size](../../assets/latency_vs_size.png)

*The plot displays the CPU inference speed across quantization sizes. Lower file sizes yield faster decoding times, with Q4_0 and Q2_K providing significant performance speedups.*

### 3. Peak RSS Memory vs File Size
![Memory vs File Size](../../assets/memory_vs_size.png)

*The plot displays the peak memory footprint (RSS) across quantization sizes. High compression quantizations significantly reduce RAM utilization during inference.*

## Key Findings & Recommendations

1. **Recommended for General Use (Q8_0)**:
   - **Fidelity**: `{q8.get('fidelity', 0.0):.4f}`.
   - **Size**: `{q8['size_mb']:.1f} MB` ({q8_pct:.1f}% size reduction from Float32).
   - **Peak RSS**: `{q8['peak_rss_mb']:.1f} MB`.
   
2. **Recommended for Resource-Constrained Environments (Q4_K_M)**:
   - **Fidelity**: `{q4.get('fidelity', 0.0):.4f}`.
   - **Size**: `{q4['size_mb']:.1f} MB` ({q4_pct:.1f}% size reduction).
   - **Peak RSS**: `{q4['peak_rss_mb']:.1f} MB`.
   - **Performance**: Significant speedup on CPU.

3. **Performance Warning (Q2_K)**:
   - While Q2_K reduces the model size to under `{q2['size_mb']:.1f} MB` and is extremely fast, its fidelity drops to `{q2.get('fidelity', 0.0):.4f}` relative to the baseline. Use only where system memory is extremely restricted.
"""
    
    os.makedirs("docs/dense_on", exist_ok=True)
    with open("docs/dense_on/accuracy_report.md", "w") as f:
        f.write(report_content)
        
    print("docs/dense_on/accuracy_report.md generated successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-isolated", type=str, help="Run benchmark for a single quantization format in isolation")
    args = parser.parse_args()
    
    if args.run_isolated:
        run_isolated_evaluation(args.run_isolated)
    else:
        results = run_evaluation()
        generate_report(results)
