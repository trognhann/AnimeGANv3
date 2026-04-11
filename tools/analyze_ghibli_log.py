"""
AnimeGANv3 Ghibli-c1 Training Log Analysis
Parses the training log and generates comprehensive evaluation charts.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        'logs', 'AnimeGANv3_Ghibli_c1_train.log')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'logs', 'analysis_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Color scheme ---
COLORS = {
    'bg': '#0d1117',
    'card': '#161b22',
    'text': '#c9d1d9',
    'grid': '#21262d',
    'accent1': '#58a6ff',  # blue
    'accent2': '#f78166',  # orange
    'accent3': '#7ee787',  # green
    'accent4': '#d2a8ff',  # purple
    'accent5': '#ff7b72',  # red
    'accent6': '#79c0ff',  # light blue
    'accent7': '#ffa657',  # light orange
    'accent8': '#d29922',  # yellow
}

def setup_dark_style():
    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['card'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text'],
        'ytick.color': COLORS['text'],
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.4,
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'legend.facecolor': COLORS['card'],
        'legend.edgecolor': COLORS['grid'],
        'legend.fontsize': 9,
    })

def parse_log(path):
    pretrain_data = []
    gan_data = []
    
    pretrain_pattern = re.compile(
        r'Epoch:\s+(\d+),\s+Step:\s+(\d+)\s+/\s+(\d+),.*Pre_train_G_loss:\s+([\d.]+)'
    )
    gan_pattern = re.compile(
        r'Epoch:\s+(\d+),\s+Step:\s+(\d+)\s+/\s*(\d+),.*'
        r'D_loss:([\d.]+)\s+~\s+G_loss:\s+([\d.]+)\s+\|\|\s+'
        r'G_support_loss:\s+([\d.]+),\s+'
        r'g_s_loss:\s+([\d.]+),\s+'
        r'con_loss:\s+([\d.]+),\s+'
        r'rs_loss:\s+([\d.]+),\s+'
        r'sty_loss:\s+([\d.]+),\s+'
        r's22:\s+([\d.]+),\s+'
        r's33:\s+([\d.]+),\s+'
        r's44:\s+([\d.]+),\s+'
        r'color_loss:\s+([\d.]+),\s+'
        r'tv_loss:\s+([\d.]+)\s+~\s+'
        r'D_support_loss:\s+([\d.]+)\s+\|\|\s+'
        r'G_main_loss:\s+([\d.]+),\s+'
        r'g_m_loss:\s+([\d.]+),\s+'
        r'p0_loss:\s+([\d.]+),\s+'
        r'p4_loss:\s+([\d.]+),\s+'
        r'tv_loss_m:\s+([\d.]+)\s+~\s+'
        r'D_main_loss:\s+([\d.]+)'
    )
    
    with open(path, 'r') as f:
        for line in f:
            m = pretrain_pattern.match(line)
            if m:
                pretrain_data.append({
                    'epoch': int(m.group(1)),
                    'step': int(m.group(2)),
                    'total_steps': int(m.group(3)),
                    'pretrain_g_loss': float(m.group(4)),
                })
                continue
            m = gan_pattern.match(line)
            if m:
                g = m.groups()
                gan_data.append({
                    'epoch': int(g[0]),
                    'step': int(g[1]),
                    'total_steps': int(g[2]),
                    'd_loss': float(g[3]),
                    'g_loss': float(g[4]),
                    'g_support_loss': float(g[5]),
                    'g_adv_loss': float(g[6]),
                    'con_loss': float(g[7]),
                    'rs_loss': float(g[8]),
                    'sty_loss': float(g[9]),
                    's22': float(g[10]),
                    's33': float(g[11]),
                    's44': float(g[12]),
                    'color_loss': float(g[13]),
                    'tv_loss': float(g[14]),
                    'd_support_loss': float(g[15]),
                    'g_main_loss': float(g[16]),
                    'g_m_loss': float(g[17]),
                    'p0_loss': float(g[18]),
                    'p4_loss': float(g[19]),
                    'tv_loss_m': float(g[20]),
                    'd_main_loss': float(g[21]),
                })
    return pretrain_data, gan_data


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        s = weight * last + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed


def epoch_avg(data, key):
    """Compute per-epoch average for a given key."""
    from collections import defaultdict
    epoch_sums = defaultdict(lambda: [0, 0])
    for d in data:
        epoch_sums[d['epoch']][0] += d[key]
        epoch_sums[d['epoch']][1] += 1
    epochs = sorted(epoch_sums.keys())
    avgs = [epoch_sums[e][0] / epoch_sums[e][1] for e in epochs]
    return epochs, avgs


def plot_pretrain(pretrain_data):
    """Chart 1: Pre-train Generator Loss"""
    setup_dark_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    
    losses = [d['pretrain_g_loss'] for d in pretrain_data]
    steps = list(range(len(losses)))
    sm = smooth(losses, 0.95)
    
    ax.plot(steps, losses, color=COLORS['accent1'], alpha=0.15, linewidth=0.5)
    ax.plot(steps, sm, color=COLORS['accent1'], linewidth=2, label='Pre-train G Loss (EMA)')
    
    # Mark epoch boundaries
    epochs_seen = set()
    for i, d in enumerate(pretrain_data):
        if d['epoch'] not in epochs_seen and d['step'] == 0:
            ax.axvline(i, color=COLORS['grid'], linestyle='--', alpha=0.5)
            ax.text(i + 5, max(losses) * 0.95, f"E{d['epoch']}", fontsize=8, color=COLORS['text'], alpha=0.6)
            epochs_seen.add(d['epoch'])
    
    ax.set_xlabel('Step (global)')
    ax.set_ylabel('Loss')
    ax.set_title('Phase 1: Pre-training Generator Loss (Content Reconstruction)', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stats box
    stats = f"Start: {losses[0]:.3f}  ->  End: {losses[-1]:.3f}  |  Min: {min(losses):.3f}  |  Reduction: {(1-losses[-1]/losses[0])*100:.1f}%"
    ax.text(0.5, -0.12, stats, transform=ax.transAxes, ha='center', fontsize=10, 
            color=COLORS['accent3'], fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_pretrain_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 01_pretrain_loss.png")


def plot_gd_loss(gan_data):
    """Chart 2: Generator vs Discriminator Loss"""
    setup_dark_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    
    epochs_g, avg_g = epoch_avg(gan_data, 'g_loss')
    epochs_d, avg_d = epoch_avg(gan_data, 'd_loss')
    
    # Top: G and D loss per epoch
    ax1.plot(epochs_g, avg_g, color=COLORS['accent1'], linewidth=2, marker='o', markersize=3, label='G_loss (avg/epoch)')
    ax1.plot(epochs_d, avg_d, color=COLORS['accent5'], linewidth=2, marker='s', markersize=3, label='D_loss (avg/epoch)')
    ax1.fill_between(epochs_g, avg_g, alpha=0.1, color=COLORS['accent1'])
    ax1.fill_between(epochs_d, avg_d, alpha=0.1, color=COLORS['accent5'])
    
    ax1.set_ylabel('Loss')
    ax1.set_title('Phase 2: Generator vs Discriminator Loss (Per Epoch)', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: G/D Ratio
    ratios = [g / max(d, 1e-8) for g, d in zip(avg_g, avg_d)]
    ax2.plot(epochs_g, ratios, color=COLORS['accent4'], linewidth=2, marker='D', markersize=3)
    ax2.axhline(y=np.mean(ratios), color=COLORS['accent8'], linestyle='--', alpha=0.7, label=f'Mean ratio: {np.mean(ratios):.1f}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('G/D Ratio')
    ax2.set_title('G_loss / D_loss Ratio (Higher = D dominating)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_gd_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 02_gd_loss.png")


def plot_sub_losses(gan_data):
    """Chart 3: Individual Generator Sub-Losses"""
    setup_dark_style()
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    loss_configs = [
        ('con_loss', 'Content Loss', COLORS['accent1'], 'Structural fidelity'),
        ('sty_loss', 'Style Loss', COLORS['accent4'], 'Ghibli aesthetic capture'),
        ('color_loss', 'Color Loss (Lab)', COLORS['accent3'], 'Color preservation'),
        ('rs_loss', 'Region Smoothing', COLORS['accent7'], 'Cel-shading smoothness'),
        ('g_adv_loss', 'Adversarial (G_adv)', COLORS['accent5'], 'Fooling discriminator'),
        ('g_main_loss', 'G Main Loss', COLORS['accent6'], 'Main generator branch'),
    ]
    
    for ax, (key, title, color, desc) in zip(axes.flatten(), loss_configs):
        epochs, avgs = epoch_avg(gan_data, key)
        ax.plot(epochs, avgs, color=color, linewidth=2, marker='o', markersize=2)
        ax.fill_between(epochs, avgs, alpha=0.15, color=color)
        ax.set_title(f'{title}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, desc, transform=ax.transAxes, fontsize=8, va='top',
                color=color, alpha=0.8)
        # Start and end annotation
        ax.annotate(f'{avgs[0]:.3f}', (epochs[0], avgs[0]), fontsize=7, color=COLORS['text'], alpha=0.7)
        ax.annotate(f'{avgs[-1]:.3f}', (epochs[-1], avgs[-1]), fontsize=7, color=COLORS['text'], alpha=0.7,
                    ha='right')
    
    fig.suptitle('Generator Sub-Loss Components (Per Epoch Average)', fontweight='bold', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_sub_losses.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 03_sub_losses.png")


def plot_style_components(gan_data):
    """Chart 4: Multi-scale Style Loss Components (s22, s33, s44)"""
    setup_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    # Left: absolute values
    for key, label, color in [('s22', 'Scale 2 (shallow)', COLORS['accent3']),
                               ('s33', 'Scale 3 (mid)', COLORS['accent8']),
                               ('s44', 'Scale 4 (deep)', COLORS['accent5'])]:
        epochs, avgs = epoch_avg(gan_data, key)
        ax1.plot(epochs, avgs, color=color, linewidth=2, marker='o', markersize=2, label=label)
        ax1.fill_between(epochs, avgs, alpha=0.1, color=color)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Style Loss by Scale (Absolute)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: proportions (stacked area)
    epochs_s22, avgs_s22 = epoch_avg(gan_data, 's22')
    _, avgs_s33 = epoch_avg(gan_data, 's33')
    _, avgs_s44 = epoch_avg(gan_data, 's44')
    total = [a + b + c for a, b, c in zip(avgs_s22, avgs_s33, avgs_s44)]
    pct_s22 = [a/t*100 for a, t in zip(avgs_s22, total)]
    pct_s33 = [a/t*100 for a, t in zip(avgs_s33, total)]
    pct_s44 = [a/t*100 for a, t in zip(avgs_s44, total)]
    
    ax2.stackplot(epochs_s22, pct_s22, pct_s33, pct_s44,
                  colors=[COLORS['accent3'], COLORS['accent8'], COLORS['accent5']],
                  labels=['s22 (shallow)', 's33 (mid)', 's44 (deep)'], alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('% of Total Style Loss')
    ax2.set_title('Style Loss Proportions by Scale', fontweight='bold')
    ax2.legend(loc='center right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_style_components.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 04_style_components.png")


def plot_discriminator_detail(gan_data):
    """Chart 5: Discriminator Analysis (Support vs Main)"""
    setup_dark_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    epochs_ds, avg_ds = epoch_avg(gan_data, 'd_support_loss')
    epochs_dm, avg_dm = epoch_avg(gan_data, 'd_main_loss')
    epochs_d, avg_d = epoch_avg(gan_data, 'd_loss')
    
    ax1.plot(epochs_d, avg_d, color=COLORS['accent5'], linewidth=2.5, label='D_loss (total)', marker='o', markersize=3)
    ax1.plot(epochs_ds, avg_ds, color=COLORS['accent7'], linewidth=1.5, label='D_support', linestyle='--', marker='s', markersize=2)
    ax1.plot(epochs_dm, avg_dm, color=COLORS['accent6'], linewidth=1.5, label='D_main', linestyle='-.', marker='^', markersize=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Discriminator Loss Decomposition', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Support vs Main generator
    epochs_gs, avg_gs = epoch_avg(gan_data, 'g_support_loss')
    epochs_gm, avg_gm = epoch_avg(gan_data, 'g_main_loss')
    
    ax2.plot(epochs_gs, avg_gs, color=COLORS['accent4'], linewidth=2, label='G_support_loss', marker='o', markersize=3)
    ax2.plot(epochs_gm, avg_gm, color=COLORS['accent3'], linewidth=2, label='G_main_loss', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Generator: Support vs Main Branch', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_discriminator_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 05_discriminator_detail.png")


def plot_summary_dashboard(pretrain_data, gan_data):
    """Chart 6: Full Training Summary Dashboard"""
    setup_dark_style()
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # --- Panel 1: Full timeline (pre-train + GAN) ---
    ax1 = fig.add_subplot(gs[0, :])
    # Pre-train part
    pt_losses = [d['pretrain_g_loss'] for d in pretrain_data]
    pt_sm = smooth(pt_losses, 0.97)
    n_pt = len(pt_losses)
    ax1.plot(range(n_pt), pt_sm, color=COLORS['accent3'], linewidth=1.5, label='Pre-train G_loss')
    ax1.axvline(n_pt, color=COLORS['accent8'], linewidth=2, linestyle='--', alpha=0.8)
    ax1.text(n_pt + 50, max(pt_sm) * 0.9, '<-- GAN Training Starts', color=COLORS['accent8'], fontsize=9, fontweight='bold')
    
    # GAN part
    g_losses = [d['g_loss'] for d in gan_data]
    g_sm = smooth(g_losses, 0.99)
    ax1.plot(range(n_pt, n_pt + len(g_losses)), g_sm, color=COLORS['accent1'], linewidth=1.5, label='G_loss (GAN)')
    
    d_losses = [d['d_loss'] for d in gan_data]
    d_sm = smooth(d_losses, 0.99)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(n_pt, n_pt + len(d_losses)), d_sm, color=COLORS['accent5'], linewidth=1, alpha=0.6, label='D_loss')
    ax1_twin.set_ylabel('D_loss', color=COLORS['accent5'])
    ax1_twin.tick_params(axis='y', labelcolor=COLORS['accent5'])
    
    ax1.set_xlabel('Global Step')
    ax1.set_ylabel('G Loss')
    ax1.set_title('Complete Training Timeline: Pre-train → Adversarial', fontweight='bold', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Panel 2: Loss balance radar (last 5 epochs) ---
    ax2 = fig.add_subplot(gs[1, 0], polar=True)
    last_5 = [d for d in gan_data if d['epoch'] >= max(d['epoch'] for d in gan_data) - 4]
    loss_keys = ['con_loss', 'sty_loss', 'color_loss', 'rs_loss', 'g_adv_loss']
    loss_labels = ['Content', 'Style', 'Color', 'Smoothing', 'Adversarial']
    loss_vals = [np.mean([d[k] for d in last_5]) for k in loss_keys]
    # Normalize to max
    max_val = max(loss_vals)
    loss_norm = [v / max_val for v in loss_vals]
    
    angles = np.linspace(0, 2 * np.pi, len(loss_keys), endpoint=False).tolist()
    loss_norm += loss_norm[:1]
    angles += angles[:1]
    
    ax2.fill(angles, loss_norm, color=COLORS['accent4'], alpha=0.2)
    ax2.plot(angles, loss_norm, color=COLORS['accent4'], linewidth=2)
    ax2.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], loss_labels, fontsize=8)
    ax2.set_title('Loss Balance\n(Last 5 Epochs)', fontweight='bold', pad=20, fontsize=11)
    ax2.set_facecolor(COLORS['card'])
    
    # --- Panel 3: Key metrics table ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    first_epoch = [d for d in gan_data if d['epoch'] == min(d['epoch'] for d in gan_data)]
    last_epoch = [d for d in gan_data if d['epoch'] == max(d['epoch'] for d in gan_data)]
    
    metrics = [
        ('G_loss', f"{np.mean([d['g_loss'] for d in first_epoch]):.2f}", f"{np.mean([d['g_loss'] for d in last_epoch]):.2f}"),
        ('D_loss', f"{np.mean([d['d_loss'] for d in first_epoch]):.3f}", f"{np.mean([d['d_loss'] for d in last_epoch]):.3f}"),
        ('Style', f"{np.mean([d['sty_loss'] for d in first_epoch]):.2f}", f"{np.mean([d['sty_loss'] for d in last_epoch]):.2f}"),
        ('Content', f"{np.mean([d['con_loss'] for d in first_epoch]):.3f}", f"{np.mean([d['con_loss'] for d in last_epoch]):.3f}"),
        ('Color', f"{np.mean([d['color_loss'] for d in first_epoch]):.3f}", f"{np.mean([d['color_loss'] for d in last_epoch]):.3f}"),
        ('G/D Ratio', f"{np.mean([d['g_loss'] for d in first_epoch]) / max(np.mean([d['d_loss'] for d in first_epoch]), 1e-8):.1f}", 
                      f"{np.mean([d['g_loss'] for d in last_epoch]) / max(np.mean([d['d_loss'] for d in last_epoch]), 1e-8):.1f}"),
    ]
    
    table = ax3.table(
        cellText=metrics,
        colLabels=['Metric', 'First Epoch', 'Last Epoch'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(COLORS['card'])
        cell.set_edgecolor(COLORS['grid'])
        cell.set_text_props(color=COLORS['text'])
        if row == 0:
            cell.set_facecolor(COLORS['accent1'])
            cell.set_text_props(color='white', fontweight='bold')
    ax3.set_title('Key Metrics Summary', fontweight='bold', fontsize=11, pad=15)
    
    # --- Panel 4: Training health indicators ---
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    avg_d_final = np.mean([d['d_loss'] for d in last_epoch])
    avg_g_final = np.mean([d['g_loss'] for d in last_epoch])
    gd_ratio = avg_g_final / max(avg_d_final, 1e-8)
    
    # Health assessment
    indicators = []
    # D_loss health
    if avg_d_final < 0.1:
        indicators.append(('[!] D_loss rat thap', f'{avg_d_final:.3f} - D qua manh', COLORS['accent5']))
    elif avg_d_final > 0.5:
        indicators.append(('[!] D_loss cao', f'{avg_d_final:.3f} - D yeu', COLORS['accent7']))
    else:
        indicators.append(('[OK] D_loss on dinh', f'{avg_d_final:.3f}', COLORS['accent3']))
    
    # G/D ratio
    if gd_ratio > 40:
        indicators.append(('[!] G/D ratio cao', f'{gd_ratio:.1f} - Mat can bang', COLORS['accent5']))
    elif gd_ratio < 5:
        indicators.append(('[!] G/D ratio thap', f'{gd_ratio:.1f}', COLORS['accent7']))
    else:
        indicators.append(('[OK] G/D ratio hop ly', f'{gd_ratio:.1f}', COLORS['accent3']))
    
    # Style loss convergence
    epochs_sty, avg_sty = epoch_avg(gan_data, 'sty_loss')
    sty_trend = (avg_sty[-1] - avg_sty[-5]) / max(avg_sty[-5], 1e-8) * 100
    if abs(sty_trend) < 3:
        indicators.append(('[OK] Style loss hoi tu', f'Delta={sty_trend:+.1f}%', COLORS['accent3']))
    elif sty_trend > 0:
        indicators.append(('[!] Style loss tang', f'Delta={sty_trend:+.1f}%', COLORS['accent5']))
    else:
        indicators.append(('[OK] Style loss giam', f'Delta={sty_trend:+.1f}%', COLORS['accent3']))
    
    # Content loss stability
    epochs_con, avg_con = epoch_avg(gan_data, 'con_loss')
    if avg_con[-1] < 0.25:
        indicators.append(('[OK] Content loss tot', f'{avg_con[-1]:.3f}', COLORS['accent3']))
    else:
        indicators.append(('[!] Content loss cao', f'{avg_con[-1]:.3f}', COLORS['accent7']))
    
    y_start = 0.85
    ax4.text(0.5, 0.95, 'Training Health', transform=ax4.transAxes, 
             fontsize=12, fontweight='bold', ha='center', color=COLORS['text'])
    for i, (label, value, color) in enumerate(indicators):
        y = y_start - i * 0.2
        ax4.text(0.1, y, label, transform=ax4.transAxes, fontsize=11, color=color, fontweight='bold')
        ax4.text(0.1, y - 0.08, value, transform=ax4.transAxes, fontsize=9, color=COLORS['text'], alpha=0.7)
    
    fig.suptitle('AnimeGANv3 Ghibli-c1 Training Dashboard', fontweight='bold', fontsize=16, y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, '06_summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 06_summary_dashboard.png")


if __name__ == '__main__':
    print("=" * 60)
    print("  AnimeGANv3 Ghibli-c1 Training Log Analysis")
    print("=" * 60)
    print(f"\nParsing: {LOG_PATH}")
    
    pretrain_data, gan_data = parse_log(LOG_PATH)
    
    print(f"  Pre-training logs: {len(pretrain_data)} steps ({pretrain_data[0]['epoch']}-{pretrain_data[-1]['epoch']} epochs)")
    print(f"  GAN training logs: {len(gan_data)} steps ({gan_data[0]['epoch']}-{gan_data[-1]['epoch']} epochs)")
    print(f"\nGenerating charts -> {OUTPUT_DIR}\n")
    
    plot_pretrain(pretrain_data)
    plot_gd_loss(gan_data)
    plot_sub_losses(gan_data)
    plot_style_components(gan_data)
    plot_discriminator_detail(gan_data)
    plot_summary_dashboard(pretrain_data, gan_data)
    
    print(f"\n" + "="*60)
    print(f"  All 6 charts saved to: {OUTPUT_DIR}")
    print("="*60)
