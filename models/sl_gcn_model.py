import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.preprocessing import create_adjacency_matrix

class SLGCNLayer(nn.Module):
    """SL-GCN层（Structured Learning GCN）"""
    
    def __init__(self, in_features: int, out_features: int, num_subsets: int = 3):
        super().__init__()
        
        self.num_subsets = num_subsets
        
        # 为每个子集创建权重
        self.convs = nn.ModuleList([
            nn.Conv2d(in_features, out_features, kernel_size=1)
            for _ in range(num_subsets)
        ])
        
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        
        # 可学习的重要性权重
        self.importance = nn.Parameter(torch.ones(num_subsets))
    
    def forward(self, x, adj_list):
        """
        Args:
            x: (batch_size, in_features, num_nodes, 1)
            adj_list: list of (num_nodes, num_nodes) adjacency matrices
            
        Returns:
            output: (batch_size, out_features, num_nodes, 1)
        """
        out = None
        
        for i, (conv, adj) in enumerate(zip(self.convs, adj_list)):
            # 图卷积
            # x: (batch_size, in_features, num_nodes, 1)
            # adj: (num_nodes, num_nodes)
            
            # 先做线性变换
            y = conv(x)  # (batch_size, out_features, num_nodes, 1)
            
            # 再做图卷积
            y = y.squeeze(-1)  # (batch_size, out_features, num_nodes)
            y = torch.matmul(y, adj.to(x.device))  # (batch_size, out_features, num_nodes)
            y = y.unsqueeze(-1)  # (batch_size, out_features, num_nodes, 1)
            
            # 加权求和
            if out is None:
                out = y * self.importance[i]
            else:
                out = out + y * self.importance[i]
        
        out = self.bn(out)
        out = self.relu(out)
        
        return out


class SLGCNGestureClassifier(nn.Module):
    """
    SL-GCN (Structured Learning Graph Convolutional Network)
    使用多个邻接矩阵子集来捕获不同的空间关系
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 10,
                 num_subsets: int = 3):
        super().__init__()
        
        # 创建多个邻接矩阵（不同的连接策略）
        adj_base = create_adjacency_matrix()
        
        # 子集1: 原始邻接矩阵
        adj1 = adj_base
        
        # 子集2: 二阶邻接（距离为2的节点）
        adj2 = torch.FloatTensor(adj_base) @ torch.FloatTensor(adj_base)
        adj2 = (adj2 > 0).float()
        adj2 = adj2 / (adj2.sum(dim=1, keepdim=True) + 1e-6)
        
        # 子集3: 自适应（初始化为均匀分布）
        adj3 = torch.ones_like(torch.FloatTensor(adj_base)) / 21
        
        self.adj_list = [
            torch.FloatTensor(adj1),
            adj2,
            adj3
        ]
        
        for i, adj in enumerate(self.adj_list):
            self.register_buffer(f'adj{i}', adj)
        
        # SL-GCN层
        self.slgcn1 = SLGCNLayer(in_channels, 64, num_subsets)
        self.slgcn2 = SLGCNLayer(64, 128, num_subsets)
        self.slgcn3 = SLGCNLayer(128, 256, num_subsets)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: shape (batch_size, 63)
            
        Returns:
            output: shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # 重塑为图结构: (batch_size, 63) -> (batch_size, 3, 21, 1)
        x = x.view(batch_size, 21, 3)
        x = x.permute(0, 2, 1).unsqueeze(-1)  # (batch_size, 3, 21, 1)
        
        # 获取邻接矩阵列表
        adj_list = [getattr(self, f'adj{i}') for i in range(len(self.adj_list))]
        
        # SL-GCN层
        x = self.slgcn1(x, adj_list)
        x = self.slgcn2(x, adj_list)
        x = self.slgcn3(x, adj_list)
        
        # 全局池化
        x = self.global_pool(x)  # (batch_size, 256, 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, 256)
        
        # 分类
        output = self.fc(x)
        
        return output