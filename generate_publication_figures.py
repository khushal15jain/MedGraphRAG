import json
import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import pi

modes = [
    ('baseline', 'Baseline'),
    ('no_graph', 'No Graph'),
    ('no_bm25', 'No BM25'),
    ('no_reranker', 'No Reranker'),
    ('dense_only', 'Dense Only')
]

metrics_to_test = [
    'Retrieval Accuracy',
    'Precision@5',
    'Recall@5',
    'Faithfulness',
    'Answer Relevance',
    'Groundedness',
    'Hallucination',
    'Explainability',
    'Clinical Reliability',
    'Latency'
]

def load_data():
    results = {}
    for fname, label in modes:
        path = f'ablation_{fname}.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                results[fname] = data.get('evaluations', [])
        else:
            print(f"Warning: {path} not found.")
            results[fname] = []
    return results

def compute_stats_and_tests(results):
    baseline_data = results['baseline']
    if not baseline_data:
        return {}
        
    stats_dict = {}
    for metric in metrics_to_test:
        stats_dict[metric] = {}
        
        # Extract baseline array
        base_arr = []
        for row in baseline_data:
            val = row.get(metric)
            if val is not None and val != "N/A" and val != "":
                base_arr.append(float(val))
        
        stats_dict[metric]['baseline'] = {
            'mean': np.mean(base_arr) if base_arr else 0,
            'std': np.std(base_arr) if base_arr else 0,
            'p_value': None # N/A for baseline vs itself
        }
        
        for fname, label in modes:
            if fname == 'baseline':
                continue
                
            test_data = results.get(fname, [])
            test_arr = []
            
            # Paired arrays
            paired_base = []
            paired_test = []
            for i in range(min(len(baseline_data), len(test_data))):
                b_val = baseline_data[i].get(metric)
                t_val = test_data[i].get(metric)
                if b_val is not None and b_val != "N/A" and b_val != "" and t_val is not None and t_val != "N/A" and t_val != "":
                    paired_base.append(float(b_val))
                    paired_test.append(float(t_val))
                    test_arr.append(float(t_val))
            
            mean = np.mean(test_arr) if test_arr else 0
            std = np.std(test_arr) if test_arr else 0
            
            p_val = 1.0
            if len(paired_base) > 1:
                try:
                    if metric == 'Latency':
                        stat, p_val = stats.ttest_rel(paired_test, paired_base)
                    else:
                        stat, p_val = stats.wilcoxon(paired_test, paired_base)
                except Exception:
                    p_val = 1.0
                    
            stats_dict[metric][fname] = {
                'mean': mean,
                'std': std,
                'p_value': p_val
            }
            
    return stats_dict

def plot_bar_chart(metric, stats_dict, filename):
    labels = [label for fname, label in modes if fname in stats_dict[metric]]
    means = [stats_dict[metric][fname]['mean'] for fname, label in modes if fname in stats_dict[metric]]
    stds = [stats_dict[metric][fname]['std'] for fname, label in modes if fname in stats_dict[metric]]
    p_vals = [stats_dict[metric][fname]['p_value'] for fname, label in modes if fname in stats_dict[metric]]
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#1f77b4' if fname == 'baseline' else '#aec7e8' for fname, label in modes if fname in stats_dict[metric]]
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor='black')
    
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Ablation Study: {metric}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add significance stars (* p<0.05, ** p<0.01, *** p<0.001)
    for i, bar in enumerate(bars):
        if i == 0: continue # Skip baseline
        p = p_vals[i]
        if p is not None and p < 0.05:
            text = '*'
            if p < 0.01: text = '**'
            if p < 0.001: text = '***'
            
            y_pos = bar.get_height() + stds[i] + (max(means)*0.02 if max(means) > 0 else 0.02)
            ax.text(bar.get_x() + bar.get_width()/2, y_pos, text, ha='center', va='bottom', fontweight='bold', color='red', fontsize=12)
            
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_radar_chart(stats_dict, filename):
    radar_metrics = ['Retrieval Accuracy', 'Faithfulness', 'Groundedness', 'Clinical Reliability']
    
    N = len(radar_metrics)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], radar_metrics, fontsize=11, fontweight='bold')
    
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=8)
    plt.ylim(0, 1)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (fname, label) in enumerate(modes):
        if fname not in stats_dict[radar_metrics[0]]:
            continue
            
        values = [stats_dict[m][fname]['mean'] for m in radar_metrics]
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2.5, linestyle='solid', label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.12, color=colors[i])
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.15, 0.15), fontsize=10)
    plt.title('MedGraphRAG Multi-Dimensional Metric Profile', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def print_markdown_table(stats_dict):
    print("| Metric | Baseline | No Graph | No BM25 | No Reranker | Dense Only |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for metric in metrics_to_test:
        row = [f"**{metric}**"]
        for fname, label in modes:
            if fname in stats_dict[metric]:
                mean = stats_dict[metric][fname]['mean']
                std = stats_dict[metric][fname]['std']
                p_val = stats_dict[metric][fname]['p_value']
                
                sig = ""
                if p_val is not None and p_val < 0.05:
                    sig = " *"
                    if p_val < 0.01: sig = " **"
                    if p_val < 0.001: sig = " ***"
                    
                row.append(f"{mean:.4f} ± {std:.4f}{sig}")
            else:
                row.append("N/A")
        print("| " + " | ".join(row) + " |")

def main():
    import shutil
    os.environ['MPLCONFIGDIR'] = os.getcwd() + '/.matplotlib'
    os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
    
    results = load_data()
    stats_dict = compute_stats_and_tests(results)
    
    if not stats_dict:
        print("No data available yet.")
        return
        
    artifact_dir = "/Users/khushaljain/.gemini/antigravity/brain/08d67381-7f1d-4787-afc6-8ffb35978b8a"
    
    for metric in ['Retrieval Accuracy', 'Faithfulness', 'Groundedness', 'Hallucination', 'Clinical Reliability', 'Latency']:
        safe_name = metric.lower().replace(' ', '_').replace('@', '_')
        fname = f'{safe_name}_chart.png'
        plot_bar_chart(metric, stats_dict, fname)
        if os.path.exists(artifact_dir):
            shutil.copy(fname, os.path.join(artifact_dir, fname))
        print(f"Generated and synced {fname}")
        
    radar_name = 'radar_chart.png'
    plot_radar_chart(stats_dict, radar_name)
    if os.path.exists(artifact_dir):
        shutil.copy(radar_name, os.path.join(artifact_dir, radar_name))
    print(f"Generated and synced {radar_name}\n")
    
    print_markdown_table(stats_dict)

if __name__ == "__main__":
    main()
