import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import json
from pathlib import Path

from models.lstm_model import LSTMGestureClassifier
from models.transformer_model import TransformerGestureClassifier
from models.st_gcn_model import STGCNGestureClassifier
from models.sl_gcn_model import SLGCNGestureClassifier
from models.mlp_model import MLPGestureClassifier
from utils.data_loader import get_dataloaders
from utils.metrics import MetricsCalculator


class Trainer:
    """统一的训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 num_classes: int,
                 class_names: list,
                 device: str = 'cuda',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 save_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 model_name: str = 'model'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.model_name = model_name
        
        # 优化器和损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )
        
        # 保存路径
        self.save_dir = Path(save_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{model_name}')
        
        # 最佳模型追踪
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int):
        """验证"""
        self.model.eval()
        
        running_loss = 0.0
        metrics_calc = MetricsCalculator(self.num_classes, self.class_names)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 前向传播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                
                # 更新指标
                probs = torch.softmax(outputs, dim=1)
                metrics_calc.update(predicted, labels, probs)
        
        epoch_loss = running_loss / len(self.val_loader)
        metrics = metrics_calc.compute()
        
        return epoch_loss, metrics, metrics_calc
    
    def train(self, num_epochs: int, early_stopping_patience: int = 20):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics, metrics_calc = self.validate(epoch)
            val_acc = val_metrics['accuracy'] * 100
            
            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            for key, value in val_metrics.items():
                if key.startswith(('precision', 'recall', 'f1')):
                    self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            
            # 打印结果
            print(f'\nEpoch {epoch}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            print(f'Val F1 (macro): {val_metrics["f1_macro"]:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                
                self.save_checkpoint(epoch, val_acc, val_metrics, is_best=True)
                print(f'✓ Best model saved! (Val Acc: {val_acc:.2f}%)')
                
                # 保存混淆矩阵
                metrics_calc.plot_confusion_matrix(
                    save_path=self.save_dir / 'best_confusion_matrix.png'
                )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                break
            
            # 定期保存检查点
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_acc, val_metrics, is_best=False)
        
        # 训练结束
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% at Epoch {self.best_epoch}")
        print(f"{'='*60}\n")
        
        # 保存训练历史
        self.save_history()
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, val_acc: float, 
                       val_metrics: dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'val_metrics': val_metrics,
            'history': self.history
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, path)
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
    
    def save_history(self):
        """保存训练历史"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


def get_model(model_name: str, num_classes: int, device: str):
    """根据名称创建模型"""
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
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available models: {list(model_dict.keys())}")
    
    model = model_dict[model_name]
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{model_name.upper()} Model:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition models')
    
    # 数据参数
    parser.add_argument('--train_dir', type=str, required=True,
                       help='Training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                       help='Validation data directory')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Test data directory (optional)')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'st_gcn', 'sl_gcn', 'mlp'],
                       help='Model architecture')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--use_cache', action='store_true',
                       help='Use cached preprocessed data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"\nUsing device: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    print("\nLoading data...")
    loaders = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_cache=args.use_cache
    )
    
    num_classes = loaders['num_classes']
    class_names = loaders['class_names']
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Train samples: {len(loaders['train'].dataset)}")
    print(f"Val samples: {len(loaders['val'].dataset)}")
    if 'test' in loaders:
        print(f"Test samples: {len(loaders['test'].dataset)}")
    
    # 创建模型
    print("\nCreating model...")
    model = get_model(args.model, num_classes, args.device)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        num_classes=num_classes,
        class_names=class_names,
        device=args.device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model
    )
    
    # 开始训练
    trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping
    )


if __name__ == '__main__':
    main()