"""Generate plots and visualizations for the final report."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_benchmark_comparison(results_path, out_path, env_name):
    """Plot mean ± std comparison bar chart."""
    with open(results_path) as f:
        data = json.load(f)
    algos = list(data.keys())
    means = [data[a]['mean'] for a in algos]
    stds = [data[a]['std'] for a in algos]
    cxs = [data[a]['avg_cx'] for a in algos]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    x = np.arange(len(algos))
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(algos)]
    # Mean ± std
    bars = ax1.bar(x, means, yerr=stds, capsize=8, color=colors, alpha=0.85, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in algos])
    ax1.set_ylabel('Final Mean Reward')
    ax1.set_title(f'{env_name}\nMean ± Std (lower is better for negative reward envs)')
    ax1.grid(axis='y', alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + s,
                f'{m:.1f}\n±{s:.1f}', ha='center', va='bottom', fontsize=9)
    # Complexity
    bars2 = ax2.bar(x, cxs, color=colors, alpha=0.85, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.upper() for a in algos])
    ax2.set_ylabel('Avg Network Complexity (nodes + conns)')
    ax2.set_title(f'{env_name}\nNetwork Size (smaller is better)')
    ax2.grid(axis='y', alpha=0.3)
    for bar, cx in zip(bars2, cxs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{cx:.1f}', ha='center', va='bottom', fontsize=10)
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved {out_path}")


def plot_training_curves(results_path, out_path, env_name):
    """Plot training curves: best fitness over generations, per algo."""
    # The benchmark script doesn't save history; let's read from older per-gen logs
    # For now, just plot the per-seed final values as a strip plot
    with open(results_path) as f:
        data = json.load(f)
    algos = list(data.keys())
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(algos)]
    for i, algo in enumerate(algos):
        per_seed = data[algo]['per_seed']
        finals = [p['final'] for p in per_seed]
        x = np.random.normal(i, 0.05, size=len(finals))
        ax.scatter(x, finals, color=colors[i], s=80, alpha=0.7, edgecolor='black', zorder=3)
        ax.scatter([i], [data[algo]['mean']], color=colors[i], marker='_', s=400, linewidth=3, zorder=4)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([a.upper() for a in algos])
    ax.set_ylabel('Final Mean Reward (15-seed eval)')
    ax.set_title(f'{env_name}: Per-seed results (bar = mean)')
    ax.grid(axis='y', alpha=0.3)
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved {out_path}")


def plot_ablation(results_path, out_path):
    """Plot ablation results as horizontal bar chart."""
    with open(results_path) as f:
        data = json.load(f)
    names = list(data.keys())
    means = [data[n]['mean'] for n in names]
    stds = [data[n]['std'] for n in names]
    # Sort by mean descending (better is higher mean for ablation results)
    sorted_idx = np.argsort(means)[::-1]
    names = [names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    y = np.arange(len(names))
    colors = ['#2E86AB' if 'full' in n else '#cccccc' for n in names]
    ax.barh(y, means, xerr=stds, capsize=4, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Final Reward')
    ax.set_title('Ablation Study on LunarLander-v3\n(blue = full CRIT, gray = ablated variants)')
    ax.grid(axis='x', alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 1, i, f'{m:.1f}', va='center', fontsize=9)
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == '__main__':
    out_dir = Path('results/plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Plot final benchmarks
    for env_tag, env_name in [('cartpole', 'CartPole-v1'),
                              ('mountaincar', 'MountainCar-v0'),
                              ('acrobot', 'Acrobot-v1'),
                              ('lunar', 'LunarLander-v3')]:
        path = f'results/benchmark_final_{env_tag}.json'
        if os.path.exists(path):
            plot_benchmark_comparison(path, f'results/plots/benchmark_{env_tag}.png', env_name)
            plot_training_curves(path, f'results/plots/perseed_{env_tag}.png', env_name)
    # Plot ablation
    if os.path.exists('results/ablation_lunar.json'):
        plot_ablation('results/ablation_lunar.json', 'results/plots/ablation_lunar.png')
    print("All plots generated.")
