import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerGestureClassifier(nn.Module):
    """
    Transformer-based手势分类器
    参考OpenHands论文
    """
    
    def __init__(self,
                 input_dim: int = 63,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_classes: int = 10):
        super().__init__()
        
        self.seq_len = 21  # 21个关键点作为序列
        self.feature_dim = 3  # 每个关键点3个坐标
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(self.feature_dim, d_model)
        
        # 位置编码 - 修改这里，max_len需要包含CLS token
        self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + 1, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # CLS token（用于分类）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: shape (batch_size, 63)
            
        Returns:
            output: shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # 重塑为序列: (batch_size, 63) -> (batch_size, 21, 3)
        x = x.view(batch_size, self.seq_len, self.feature_dim)
        
        # 输入投影
        x = self.input_projection(x)  # (batch_size, 21, d_model)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 22, d_model)
        
        # 位置编码（需要转换维度以匹配PositionalEncoding的期望输入）
        # PositionalEncoding expects: (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)  # (22, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, 22, d_model)
        
        # Transformer Encoder
        x = self.transformer_encoder(x)
        
        # 使用CLS token进行分类
        cls_output = x[:, 0, :]  # (batch_size, d_model)
        
        # 分类
        output = self.classifier(cls_output)
        
        return output