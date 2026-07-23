"""
Mamba-Long 可解释性可视化 - 步骤 4

从 outputs/mamba_long/{event_importance, temporal_importance, proto_weights, micro_repr}.npy
生成 4 类可解释性图:
  1. 事件重要性热力图 (7 种事件 × 473 学生)
  2. 时间窗口重要性热力图 (100 窗口 × 473 学生)
  3. 4 类行为原型分布 (饼图 + 平均条形图)
  4. Micro 特征 t-SNE 投影 (12d → 2d)
  5. 关键学生风险时间轴 (top 10 风险学生)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

OUT = "/home/ubuntu/CodeEMO/outputs/mamba_long"
FIG_DIR = os.path.join(OUT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("  Mamba-Long 可解释性可视化 (步骤 4)")
    print("=" * 70)

    # 加载
    probs = np.load(os.path.join(OUT, 'probs.npy'))
    y = np.load(os.path.join(OUT, 'labels.npy'))
    event_imp = np.load(os.path.join(OUT, 'event_importance.npy'))
    temporal_imp = np.load(os.path.join(OUT, 'temporal_importance.npy'))
    proto_w = np.load(os.path.join(OUT, 'proto_weights.npy'))
    micro_repr = np.load(os.path.join(OUT, 'micro_repr.npy'))
    with open(os.path.join(OUT, 'event_names.json')) as f:
        event_names = json.load(f)

    print(f"  样本: {len(y)}, fail_rate: {y.mean():.3f}")
    print(f"  event_importance: {event_imp.shape}")
    print(f"  temporal_importance: {temporal_imp.shape}")
    print(f"  proto_weights: {proto_w.shape}")
    print(f"  micro_repr: {micro_repr.shape}")

    # ===== 1. 事件重要性热力图 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    # 按 fail/pass 分组显示
    imp_fail = event_imp[y == 1].mean(axis=0)
    imp_pass = event_imp[y == 0].mean(axis=0)
    x = np.arange(len(event_names))
    w = 0.35
    ax.bar(x - w/2, imp_pass, w, label='Pass (correct)', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, imp_fail, w, label='Fail (at-risk)', color='indianred', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(event_names, rotation=30, ha='right')
    ax.set_ylabel('Avg importance')
    ax.set_title('Event Importance: Pass vs Fail students')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 整体热力图
    ax = axes[1]
    # 抽样 50 fail + 50 pass 展示
    idx_fail = np.where(y == 1)[0][:50]
    idx_pass = np.where(y == 0)[0][:50]
    sample = np.concatenate([idx_pass, idx_fail])
    im = ax.imshow(event_imp[sample], aspect='auto', cmap='YlOrRd')
    ax.set_xticks(np.arange(len(event_names)))
    ax.set_xticklabels(event_names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Student (top 50 pass + 50 fail)')
    ax.set_title('Event Importance Heatmap')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out1 = os.path.join(FIG_DIR, '01_event_importance.png')
    plt.savefig(out1, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 {out1}")

    # ===== 2. 时间窗口重要性 =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    n_win = temporal_imp.shape[1]
    ti_fail = temporal_imp[y == 1].mean(axis=0)
    ti_pass = temporal_imp[y == 0].mean(axis=0)
    x = np.arange(n_win)
    ax.plot(x, ti_pass, 'o-', label='Pass', color='steelblue', linewidth=2)
    ax.plot(x, ti_fail, 's-', label='Fail', color='indianred', linewidth=2)
    ax.set_xlabel('Time window (100-event window)')
    ax.set_ylabel('Avg importance')
    ax.set_title('Temporal Importance: when are risk events happening?')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    diff = ti_fail - ti_pass
    colors = ['indianred' if d > 0 else 'steelblue' for d in diff]
    ax.bar(x, diff, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Time window')
    ax.set_ylabel('Fail - Pass importance')
    ax.set_title('Risk Contribution per Window (red = fail-heavy, blue = pass-heavy)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, '02_temporal_importance.png')
    plt.savefig(out2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 {out2}")

    # ===== 3. 原型分布 =====
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 3a. 平均原型占比
    ax = axes[0]
    pw_fail = proto_w[y == 1].mean(axis=0)
    pw_pass = proto_w[y == 0].mean(axis=0)
    x = np.arange(proto_w.shape[1])
    w = 0.35
    ax.bar(x - w/2, pw_pass, w, label='Pass', color='steelblue')
    ax.bar(x + w/2, pw_fail, w, label='Fail', color='indianred')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i}' for i in range(proto_w.shape[1])])
    ax.set_ylabel('Avg prototype weight')
    ax.set_title('Behavior Prototype Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3b. 整体饼图
    ax = axes[1]
    proto_id = proto_w.argmax(axis=1)
    counts = [np.sum(proto_id == i) for i in range(proto_w.shape[1])]
    ax.pie(counts, labels=[f'P{i}\n({c})' for i, c in enumerate(counts)],
           autopct='%1.1f%%', colors=plt.cm.Set2.colors[:4])
    ax.set_title('Overall Prototype Population')

    # 3c. Fail vs Pass prototype 占比
    ax = axes[2]
    fail_pids = [np.sum(proto_id[y == 1] == i) / max(np.sum(y == 1), 1)
                 for i in range(proto_w.shape[1])]
    pass_pids = [np.sum(proto_id[y == 0] == i) / max(np.sum(y == 0), 1)
                 for i in range(proto_w.shape[1])]
    x = np.arange(proto_w.shape[1])
    ax.bar(x - w/2, pass_pids, w, label='Pass', color='steelblue')
    ax.bar(x + w/2, fail_pids, w, label='Fail', color='indianred')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i}' for i in range(proto_w.shape[1])])
    ax.set_ylabel('Prototype share')
    ax.set_title('Prototype Share by Outcome')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out3 = os.path.join(FIG_DIR, '03_prototype_distribution.png')
    plt.savefig(out3, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 {out3}")

    # ===== 4. Micro 特征 t-SNE =====
    print(f"  t-SNE 投影 micro_repr {micro_repr.shape} ...")
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        micro_2d = tsne.fit_transform(micro_repr)
    except Exception as e:
        print(f"  t-SNE 失败, 改用 PCA: {e}")
        from sklearn.decomposition import PCA
        micro_2d = PCA(n_components=2).fit_transform(micro_repr)

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, color, name in [(1, 'indianred', 'Fail'), (0, 'steelblue', 'Pass')]:
        mask = y == label
        ax.scatter(micro_2d[mask, 0], micro_2d[mask, 1],
                   c=color, label=name, alpha=0.6, s=30)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.set_title('Micro Feature Space (12d → 2d, colored by outcome)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out4 = os.path.join(FIG_DIR, '04_micro_tsne.png')
    plt.savefig(out4, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 {out4}")

    # ===== 5. Top 10 风险学生时间轴 =====
    top_k = 10
    top_idx = np.argsort(probs)[-top_k:][::-1]  # 高风险学生

    fig, axes = plt.subplots(top_k, 1, figsize=(14, 2 * top_k), sharex=True)
    if top_k == 1:
        axes = [axes]
    for i, idx in enumerate(top_idx):
        ax = axes[i]
        ti = temporal_imp[idx]
        ax.fill_between(np.arange(len(ti)), ti, alpha=0.4, color='indianred')
        ax.plot(ti, color='indianred', linewidth=1.5)
        risk_prob = probs[idx]
        actual = 'Fail' if y[idx] == 1 else 'Pass'
        ax.set_title(f"Student #{idx}: P(fail)={risk_prob:.3f}, actual={actual}",
                     fontsize=10, loc='left')
        ax.set_ylabel('Risk signal', fontsize=9)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time window (100-event window from early to late)')
    plt.tight_layout()
    out5 = os.path.join(FIG_DIR, '05_top10_risk_timeline.png')
    plt.savefig(out5, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 {out5}")

    # ===== Markdown 报告 =====
    md = ["# Mamba-Long 可解释性报告 (步骤 4)\n",
          f"## 数据概览\n- 样本: {len(y)}\n- 失败率: {y.mean():.3f}\n",
          "## 关键发现\n"]

    # 事件重要性 - 哪个事件对 fail 学生最关键
    fail_only_imp = event_imp[y == 1].mean(axis=0)
    top_event_idx = np.argsort(fail_only_imp)[-3:][::-1]
    md.append("### 1. 事件重要性 (Fail 学生)")
    for idx in top_event_idx:
        md.append(f"- **{event_names[idx]}**: 重要性={fail_only_imp[idx]:.4f}")
    md.append("")

    # 时间窗口 - 哪个时段最关键
    diff = ti_fail - ti_pass
    top_window_idx = np.argsort(diff)[-3:][::-1]
    md.append("### 2. 危险时段 (Fail 比 Pass 重要性更高的窗口)")
    for idx in top_window_idx:
        md.append(f"- Window #{idx}: Δ importance = {diff[idx]:+.4f}")
    md.append("")

    # 原型 - fail 学生主要聚类
    fail_pids = [np.sum(proto_id[y == 1] == i) / max(np.sum(y == 1), 1)
                 for i in range(proto_w.shape[1])]
    top_proto_idx = np.argsort(fail_pids)[-2:][::-1]
    md.append("### 3. 主导行为原型 (Fail 学生)")
    for idx in top_proto_idx:
        md.append(f"- Prototype #{idx}: {fail_pids[idx]*100:.1f}% 的 fail 学生属于此原型")
    md.append("")

    md.append("## 可视化\n"
              f"- 事件重要性: `figures/01_event_importance.png`\n"
              f"- 时间窗口: `figures/02_temporal_importance.png`\n"
              f"- 原型分布: `figures/03_prototype_distribution.png`\n"
              f"- Micro t-SNE: `figures/04_micro_tsne.png`\n"
              f"- Top10 风险时间轴: `figures/05_top10_risk_timeline.png`\n")

    md_path = os.path.join(OUT, 'interpret_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print(f"\n  📄 {md_path}")
    print(f"\n{'='*70}\n  全部完成, 共 5 张图 + Markdown 报告\n{'='*70}")


if __name__ == '__main__':
    main()