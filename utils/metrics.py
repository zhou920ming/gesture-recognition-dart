import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

class MetricsCalculator:
    """计算和记录各种评估指标"""
    
    def __init__(self, num_classes: int, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """重置所有指标"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor, 
               probs: torch.Tensor = None):
        """
        更新指标
        
        Args:
            predictions: 预测的类别
            labels: 真实标签
            probs: 预测概率（可选）
        """
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy())
    
    def compute(self) -> dict:
        """计算所有指标"""
        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'precision_weighted': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall_weighted': recall_score(labels, predictions, average='weighted', zero_division=0),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
        }
        
        # 每个类别的指标
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        
        for i in range(self.num_classes):
            metrics[f'precision_class_{i}'] = precision_per_class[i]
            metrics[f'recall_class_{i}'] = recall_per_class[i]
            metrics[f'f1_class_{i}'] = f1_per_class[i]
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: str = None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(self.all_labels, self.all_predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()