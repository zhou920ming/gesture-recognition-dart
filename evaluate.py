import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.lstm_model import LSTMGestureClassifier
from models.transformer_model import TransformerGestureClassifier
from models.st_gcn_model import STGCNGestureClassifier
from models.sl_gcn_model import SLGCNGestureClassifier
from models.mlp_model import MLPGestureClassifier
from utils.data_loader import get_dataloaders
from utils.metrics import MetricsCalculator


def load_model(model_name: str, checkpoint_path: str, 
               num_classes: int, device: str):
    """加载训练好的模型"""
    
    # 创建模型
    model_dict = {
        'lstm': LSTMGestureClassifier(
            input_dim=63,
            hidden_dim=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.5,
            bidirectional=True
        ),
        'transformer': TransformerGestureClassifier(
            input_dim=63,
            d_model=128,
            nhead=8,
            num_encoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            num_classes=num_classes
        ),
        'st_gcn': STGCNGestureClassifier(
            in_channels=3,
            num_classes=num_classes
        ),
        'sl_gcn': SLGCNGestureClassifier(
            in_channels=3,
            num_classes=num_classes,
            num_subsets=3
        ),
        'mlp': MLPGestureClassifier(
            input_dim=63,
            hidden_dims=[256, 128, 64],
            num_classes=num_classes,
            dropout=0.5
        )
    }
    
    model = model_dict[model_name].to(device)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model, checkpoint


def evaluate_model(model: nn.Module,
                   data_loader,
                   num_classes: int,
                   class_names: list,
                   device: str,
                   save_dir: Path):
    """评估模型"""
    
    model.eval()
    metrics_calc = MetricsCalculator(num_classes, class_names)
    
    all_outputs = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            probs = torch.softmax(outputs, dim=1)
            
            # 更新指标
            metrics_calc.update(predicted, labels, probs)
            all_outputs.append(outputs.cpu())
    
    # 计算指标
    metrics = metrics_calc.compute()
    
    # 打印结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
    
    print("\nPer-class metrics:")
    print("-"*60)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:20s} | "
              f"Precision: {metrics[f'precision_class_{i}']:.4f} | "
              f"Recall: {metrics[f'recall_class_{i}']:.4f} | "
              f"F1: {metrics[f'f1_class_{i}']:.4f}")
    
    # 保存结果
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存指标到JSON
    with open(save_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # 绘制混淆矩阵
    metrics_calc.plot_confusion_matrix(save_dir / 'confusion_matrix.png')
    
    # 绘制每个类别的性能
    plot_per_class_metrics(metrics, class_names, save_dir)
    
    return metrics


def plot_per_class_metrics(metrics: dict, class_names: list, save_dir: Path):
    """绘制每个类别的指标"""
    num_classes = len(class_names)
    
    precision = [metrics[f'precision_class_{i}'] for i in range(num_classes)]
    recall = [metrics[f'recall_class_{i}'] for i in range(num_classes)]
    f1 = [metrics[f'f1_class_{i}'] for i in range(num_classes)]
    
    x = np.arange(num_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(12, num_classes * 0.8), 6))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(history_path: Path, save_dir: Path):
    """绘制训练历史"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Accuracy曲线
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(alpha=0.3)
    
    # Overfitting分析
    gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(gap)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Train Acc - Val Acc (%)')
    axes[1, 1].set_title('Overfitting Analysis')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate gesture recognition models')
    
    # 模型参数
    parser.add_argument('--model', type=str, required=True,
                       choices=['lstm', 'transformer', 'st_gcn', 'sl_gcn', 'mlp'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Data directory for evaluation')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--use_cache', action='store_true',
                       help='Use cached preprocessed data')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # 加载数据
    print("\nLoading data...")
    from utils.data_loader import GestureDataset
    from torch.utils.data import DataLoader
    
    dataset = GestureDataset(
        args.data_dir,
        cache_file=f'./cache/eval_cache.pkl' if args.use_cache else None
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    num_classes = len(dataset.class_names)
    class_names = dataset.class_names
    
    print(f"Number of classes: {num_classes}")
    print(f"Number of samples: {len(dataset)}")
    
    # 加载模型
    print("\nLoading model...")
    model, checkpoint = load_model(
        args.model,
        args.checkpoint,
        num_classes,
        args.device
    )
    
    # 评估
    save_dir = Path(args.save_dir) / args.model
    metrics = evaluate_model(
        model,
        data_loader,
        num_classes,
        class_names,
        args.device,
        save_dir
    )
    
    # 绘制训练历史（如果存在）
    checkpoint_dir = Path(args.checkpoint).parent
    history_path = checkpoint_dir / 'training_history.json'
    if history_path.exists():
        print("\nPlotting training history...")
        plot_training_history(history_path, save_dir)
    
    print(f"\nEvaluation results saved to {save_dir}")


if __name__ == '__main__':
    main()