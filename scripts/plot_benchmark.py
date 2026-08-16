import os
import matplotlib.pyplot as plt
import numpy as np

def apply_transparent_style(ax):
    text_color = '#334155'
    tick_color = '#64748b'
    grid_color = '#cbd5e1'
    
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color(tick_color)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color(tick_color)
    
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    
    ax.tick_params(colors=tick_color, which='both')
    ax.grid(True, linestyle=':', color=grid_color, alpha=0.3)

def plot_bar_3(title, val_ggmbed, val_fastembed, val_st, ylabel, filename, val_format='{:.1f}'):
    fig, ax = plt.subplots(figsize=(5.2, 3.8), facecolor='none')
    
    color_ggmbed = '#4f46e5'
    color_fastembed = '#0284c7'
    color_st = '#94a3b8'
    
    width = 0.25
    rects1 = ax.bar([-0.28], [val_ggmbed], width, label='ggmbed (GGUF)', color=color_ggmbed)
    rects2 = ax.bar([0.0], [val_fastembed], width, label='fastembed (ONNX)', color=color_fastembed)
    rects3 = ax.bar([0.28], [val_st], width, label='sentence-transformers', color=color_st)
    
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.set_xlim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    legend = ax.legend(frameon=True, facecolor='none', edgecolor='none', fontsize=8, loc='upper right')
    for text in legend.get_texts():
        text.set_color('#334155')
        
    apply_transparent_style(ax)
    
    for rect, color in [(rects1, color_ggmbed), (rects2, color_fastembed), (rects3, color_st)]:
        for r in rect:
            height = r.get_height()
            ax.annotate(val_format.format(height), xy=(r.get_x() + r.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                        fontsize=8.5, fontweight='bold', color=color)
                        
    plt.tight_layout()
    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()

def plot_line_3(title, batch_sizes, y_ggmbed, y_fastembed, y_st, filename):
    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor='none')
    
    ax.plot(batch_sizes, y_ggmbed, marker='o', linewidth=2.5, label='ggmbed (GGUF)', color='#4f46e5')
    ax.plot(batch_sizes, y_fastembed, marker='s', linewidth=2, label='fastembed (ONNX)', color='#0284c7')
    ax.plot(batch_sizes, y_st, marker='^', linestyle='--', linewidth=2, label='sentence-transformers', color='#94a3b8')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.set_xlabel('Batch Size (Log Scale)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Throughput sent/sec (Log Scale)', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    legend = ax.legend(frameon=True, facecolor='none', edgecolor='none', fontsize=8)
    for text in legend.get_texts():
        text.set_color('#334155')
        
    apply_transparent_style(ax)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()

def main():
    os.makedirs('assets', exist_ok=True)
    batch_sizes = [1, 4, 8, 32, 128]
    
    # ------------------ MiniLM Charts ------------------
    plot_bar_3(
        title='MiniLM: Model Load Time\n(Lower is Better)',
        val_ggmbed=1909.7, val_fastembed=13488.1, val_st=20179.3,
        ylabel='Load Time (ms)',
        filename='assets/minilm_latency.png',
        val_format='{:.0f}ms'
    )
    
    plot_line_3(
        title='MiniLM: Throughput Scaling\n(Higher is Better)',
        batch_sizes=batch_sizes,
        y_ggmbed=[78.8, 83.4, 79.1, 78.7, 79.0],
        y_fastembed=[86.1, 96.3, 65.3, 55.0, 37.2],
        y_st=[55.1, 348.3, 1469.5, 4608.6, 6053.3],
        filename='assets/minilm_throughput.png'
    )
    
    plot_bar_3(
        title='MiniLM: Peak Memory Footprint\n(Lower is Better)',
        val_ggmbed=127.6, val_fastembed=910.7, val_st=785.0,
        ylabel='Peak RAM (MB)',
        filename='assets/minilm_memory.png',
        val_format='{:.1f}MB'
    )
    
    # ------------------ BGE Charts ------------------
    plot_bar_3(
        title='BGE-Small: Single Latency\n(Lower is Better)',
        val_ggmbed=6.73, val_fastembed=9.58, val_st=6.14,
        ylabel='Latency (ms)',
        filename='assets/bge_latency.png',
        val_format='{:.2f}ms'
    )
    
    plot_line_3(
        title='BGE-Small: Throughput Scaling\n(Higher is Better)',
        batch_sizes=batch_sizes,
        y_ggmbed=[268.0, 220.5, 206.6, 199.7, 198.5],
        y_fastembed=[130.2, 252.4, 292.2, 286.1, 158.1],
        y_st=[66.1, 230.7, 949.7, 2979.9, 3496.5],
        filename='assets/bge_throughput.png'
    )
    
    plot_bar_3(
        title='BGE-Small: Peak Memory Footprint\n(Lower is Better)',
        val_ggmbed=111.0, val_fastembed=351.4, val_st=806.2,
        ylabel='Peak RAM (MB)',
        filename='assets/bge_memory.png',
        val_format='{:.1f}MB'
    )
    
    print("Generated all transparent benchmark charts in assets/")

if __name__ == '__main__':
    main()
