"""
Visualization Module for Mamba CS1 Experiment Results
====================================================

Generates visualizations for:
1. Model performance metrics (5-fold CV)
2. Prototype cluster distribution
3. Event type importance
4. Temporal importance patterns
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
import os

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def load_results(results_file):
    """Load experiment results from JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_cv_results(results, output_dir):
    """Plot 5-fold cross-validation results"""
    fold_results = results['fold_results']
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        bars = ax.bar(range(1, 6), values, color='steelblue', alpha=0.7)
        ax.axhline(y=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
        ax.fill_between([0.5, 5.5], mean_val - std_val, mean_val + std_val,
                       color='red', alpha=0.1, label=f'±{std_val:.3f}')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(title)
        ax.set_title(f'{title} across 5 Folds')
        ax.set_xticks(range(1, 6))
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.1])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'cv_results.png'}")


def plot_metrics_comparison(results, output_dir):
    """Plot comparison of all metrics with error bars"""
    summary = results['summary']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    means = [summary[f'{m}_mean'] for m in metrics]
    stds = [summary[f'{m}_std'] for m in metrics]
    
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Mamba 7-Dim Model Performance (5-Fold CV)')
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.set_ylim([0, 1.15])
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Baseline 0.8')
    ax.legend()
    
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.03,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'metrics_comparison.png'}")


def plot_prototype_distribution(results, output_dir):
    """Plot prototype cluster distribution and risk rates"""
    prototype_info = results['prototype_info']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Cluster sizes
    clusters = [p['cluster'] for p in prototype_info]
    sizes = [p['n'] for p in prototype_info]
    risk_rates = [p['risk_rate'] for p in prototype_info]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars = axes[0].bar(clusters, sizes, color=colors, alpha=0.8)
    axes[0].set_xlabel('Prototype Cluster')
    axes[0].set_ylabel('Number of Students')
    axes[0].set_title('Student Distribution Across Prototypes')
    axes[0].set_xticks(clusters)
    
    for bar, n in zip(bars, sizes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'n={n}', ha='center', va='bottom')
    
    # Right: Risk rates by cluster
    risk_colors = ['#FF6B6B' if r > 0.7 else '#4ECDC4' if r < 0.5 else '#FFE66D' for r in risk_rates]
    bars = axes[1].bar(clusters, risk_rates, color=risk_colors, alpha=0.8)
    axes[1].set_xlabel('Prototype Cluster')
    axes[1].set_ylabel('Risk Rate')
    axes[1].set_title('Risk Rate by Prototype Cluster')
    axes[1].set_xticks(clusters)
    axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylim([0, 1])
    
    for bar, r in zip(bars, risk_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{r:.1%}', ha='center', va='bottom')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='High Risk (>70%)'),
        Patch(facecolor='#FFE66D', label='Medium Risk (50-70%)'),
        Patch(facecolor='#4ECDC4', label='Low Risk (<50%)')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prototype_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'prototype_distribution.png'}")


def plot_event_importance(results, output_dir):
    """Plot event type importance"""
    event_importance = results['event_importance']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    events = [e['event'] for e in event_importance]
    importances = [e['importance'] for e in event_importance]
    
    # Sort by importance
    sorted_idx = np.argsort(importances)
    events = [events[i] for i in sorted_idx]
    importances = [importances[i] for i in sorted_idx]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(events)))[::-1]
    
    bars = ax.barh(events, importances, color=colors, alpha=0.8)
    ax.set_xlabel('Importance Score')
    ax.set_title('Event Type Importance (Mamba Embedding Norms)')
    ax.set_xlim([0, max(importances) * 1.2])
    
    for bar, imp in zip(bars, importances):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{imp:.4f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'event_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'event_importance.png'}")


def plot_fold_comparison(results, output_dir):
    """Plot comparison across folds (radar chart)"""
    fold_results = results['fold_results']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, fold in enumerate(fold_results):
        values = [fold[m] for m in metrics]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Fold {i+1}', alpha=0.7)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0))
    ax.set_title('5-Fold Cross-Validation Comparison', size=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fold_comparison_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'fold_comparison_radar.png'}")


def generate_summary_report(results, output_dir):
    """Generate text summary report"""
    summary = results['summary']
    
    report = f"""
================================================================================
MAMBA 7-DIM EXPERIMENT RESULTS SUMMARY
================================================================================

Experiment: {results['experiment']}

Dataset:
  - Total Students: {results['dataset']['total']}
  - At-Risk (Failed): {results['dataset']['at_risk']} ({100*results['dataset']['at_risk']/results['dataset']['total']:.1f}%)
  - Passed: {results['dataset']['passed']} ({100*results['dataset']['passed']/results['dataset']['total']:.1f}%)

Model Configuration:
  - Hidden Dimension: {results.get('model_config', {}).get('d_model', 'N/A')}
  - Number of Layers: {results.get('model_config', {}).get('n_layers', 'N/A')}
  - State Dimension: {results.get('model_config', {}).get('d_state', 'N/A')}
  - Total Parameters: {results.get('model_config', {}).get('total_parameters', results.get('model_config', {}).get('total_params', 'N/A'))}

================================================================================
5-FOLD CROSS-VALIDATION RESULTS
================================================================================

  Accuracy:   {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}
  F1 Score:  {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}
  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}
  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}
  AUC:       {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}

================================================================================
PROTOTYPE CLUSTERS (4 Learning Modes)
================================================================================
"""
    
    for p in results['prototype_info']:
        risk_level = 'HIGH RISK' if p['risk_rate'] > 0.7 else 'MEDIUM RISK' if p['risk_rate'] > 0.5 else 'LOW RISK'
        report += f"  Cluster {p['cluster']}: n={p['n']}, risk_rate={p['risk_rate']:.2%} [{risk_level}]\\n"
    
    report += """
================================================================================
EVENT TYPE IMPORTANCE
================================================================================
"""
    
    for e in sorted(results['event_importance'], key=lambda x: x['importance'], reverse=True):
        report += f"  {e['event']}: {e['importance']:.4f}\\n"
    
    report += f"""
================================================================================
RUNTIME
================================================================================
  Total Runtime: {results['runtime_minutes']:.1f} minutes

================================================================================
VISUALIZATION FILES
================================================================================
  - cv_results.png: Cross-validation results by fold
  - metrics_comparison.png: Performance metrics comparison
  - prototype_distribution.png: Prototype cluster analysis
  - event_importance.png: Event type importance ranking
  - fold_comparison_radar.png: Radar chart comparison across folds

================================================================================
"""
    
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write(report)
    
    print(f"Saved: {output_dir / 'summary_report.txt'}")
    print(report)


def main():
    """Main visualization pipeline"""
    # Paths
    project_dir = Path(__file__).parent.parent
    results_file = project_dir / 'results' / 'mamba_cs1_results.json'
    output_dir = project_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("MAMBA CS1 Experiment - Visualization Pipeline")
    print("=" * 70)
    
    # Check if results exist
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print("Please run mamba_7dim_experiment.py first!")
        return
    
    # Load results
    print(f"\nLoading results from {results_file}")
    results = load_results(results_file)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_cv_results(results, output_dir)
    plot_metrics_comparison(results, output_dir)
    plot_prototype_distribution(results, output_dir)
    plot_event_importance(results, output_dir)
    plot_fold_comparison(results, output_dir)
    generate_summary_report(results, output_dir)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
