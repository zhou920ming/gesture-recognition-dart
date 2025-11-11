import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def compare_all_models(results_dir: str = './evaluation_results'):
    """对比所有模型的结果"""
    
    results_dir = Path(results_dir)
    models = ['mlp', 'lstm', 'transformer', 'st_gcn', 'sl_gcn']
    
    all_results = {}
    
    # 收集所有结果
    for model in models:
        metrics_file = results_dir / model / 'evaluation_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_results[model] = json.load(f)
    
    if not all_results:
        print("No evaluation results found!")
        return
    
    # 创建对比表格
    comparison_data = []
    for model, metrics in all_results.items():
        comparison_data.append({
            'Model': model.upper(),
            'Accuracy (%)': metrics['accuracy'] * 100,
            'Precision (macro)': metrics['precision_macro'],
            'Recall (macro)': metrics['recall_macro'],
            'F1-score (macro)': metrics['f1_macro'],
            'Precision (weighted)': metrics['precision_weighted'],
            'Recall (weighted)': metrics['recall_weighted'],
            'F1-score (weighted)': metrics['f1_weighted']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy (%)', ascending=False)
    
    # 打印表格
    print("\n" + "="*80)
    print("Model Comparison Results")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # 保存到CSV
    df.to_csv(results_dir / 'model_comparison.csv', index=False)
    
    # 绘制对比图
    plot_model_comparison(df, results_dir)
    
    return df


def plot_model_comparison(df: pd.DataFrame, save_dir: Path):
    """绘制模型对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = df['Model'].values
    x = np.arange(len(models))
    width = 0.35
    
    # Accuracy对比
    axes[0, 0].bar(x, df['Accuracy (%)'].values, color='steelblue', alpha=0.8)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Precision vs Recall (macro)
    axes[0, 1].bar(x - width/2, df['Precision (macro)'].values, 
                   width, label='Precision', alpha=0.8)
    axes[0, 1].bar(x + width/2, df['Recall (macro)'].values, 
                   width, label='Recall', alpha=0.8)
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision vs Recall (Macro)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # F1-score对比
    axes[1, 0].bar(x, df['F1-score (macro)'].values, color='coral', alpha=0.8)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('F1-score')
    axes[1, 0].set_title('F1-score (Macro) Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 综合指标雷达图
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(2, 2, 4, projection='polar')
    
    for idx, model in enumerate(models):
        values = [
            df.loc[df['Model'] == model, 'Accuracy (%)'].values[0] / 100,
            df.loc[df['Model'] == model, 'Precision (macro)'].values[0],
            df.loc[df['Model'] == model, 'Recall (macro)'].values[0],
            df.loc[df['Model'] == model, 'F1-score (macro)'].values[0]
        ]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plots saved to {save_dir / 'model_comparison.png'}")


if __name__ == '__main__':
    compare_all_models()