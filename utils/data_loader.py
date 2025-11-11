import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List
from pathlib import Path
from .preprocessing import MediaPipePreprocessor
import pickle

class GestureDataset(Dataset):
    """手势识别数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 preprocess: bool = True,
                 cache_file: str = None,
                 transform=None):
        """
        Args:
            data_dir: 数据目录，包含子文件夹，每个子文件夹是一个类别
            preprocess: 是否使用MediaPipe预处理
            cache_file: 缓存文件路径
            transform: 数据增强
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.preprocess = preprocess
        
        # 尝试从缓存加载
        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.landmarks = cache_data['landmarks']
                self.labels = cache_data['labels']
                self.class_names = cache_data['class_names']
        else:
            # 加载数据
            self.landmarks, self.labels, self.class_names = self._load_data()
            
            # 保存缓存
            if cache_file:
                print(f"Saving cache to {cache_file}")
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'landmarks': self.landmarks,
                        'labels': self.labels,
                        'class_names': self.class_names
                    }, f)
        
        print(f"Loaded {len(self.landmarks)} samples from {len(self.class_names)} classes")
    
    def _load_data(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """加载数据"""
        landmarks_list = []
        labels_list = []
        class_names = []
        
        # 获取所有类别文件夹
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if self.preprocess:
            preprocessor = MediaPipePreprocessor()
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_names.append(class_name)
            
            # 获取该类别下的所有图像
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))
            
            print(f"Processing class {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                if self.preprocess:
                    # 使用MediaPipe提取关键点
                    landmarks = preprocessor.extract_landmarks(str(img_file))
                    if landmarks is not None:
                        # 归一化
                        landmarks = preprocessor.normalize_landmarks(landmarks)
                        landmarks_list.append(landmarks)
                        labels_list.append(class_idx)
                else:
                    # 假设已经预处理好，直接加载npy文件
                    landmarks = np.load(str(img_file))
                    landmarks_list.append(landmarks)
                    labels_list.append(class_idx)
        
        if self.preprocess:
            preprocessor.close()
        
        return landmarks_list, labels_list, class_names
    
    def __len__(self) -> int:
        return len(self.landmarks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        landmarks = self.landmarks[idx]
        label = self.labels[idx]
        
        if self.transform:
            landmarks = self.transform(landmarks)
        
        return torch.FloatTensor(landmarks), label


def get_dataloaders(train_dir: str,
                   val_dir: str,
                   test_dir: str = None,
                   batch_size: int = 32,
                   num_workers: int = 4,
                   use_cache: bool = True) -> dict:
    """
    创建数据加载器
    
    Returns:
        包含train_loader, val_loader, test_loader的字典
    """
    cache_dir = './cache' if use_cache else None
    
    # 检测是否有GPU
    pin_memory = torch.cuda.is_available()
    
    # 训练集
    train_dataset = GestureDataset(
        train_dir,
        cache_file=f'{cache_dir}/train_cache.pkl' if cache_dir else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory  # 修改这里
    )
    
    # 验证集
    val_dataset = GestureDataset(
        val_dir,
        cache_file=f'{cache_dir}/val_cache.pkl' if cache_dir else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory  # 修改这里
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'num_classes': len(train_dataset.class_names),
        'class_names': train_dataset.class_names
    }
    
    # 测试集（可选）
    if test_dir:
        test_dataset = GestureDataset(
            test_dir,
            cache_file=f'{cache_dir}/test_cache.pkl' if cache_dir else None
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory  # 修改这里
        )
        loaders['test'] = test_loader
    
    return loaders