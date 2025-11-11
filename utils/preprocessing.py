import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class MediaPipePreprocessor:
    """使用MediaPipe提取手部关键点"""
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def extract_landmarks(self, image_path: str) -> Optional[np.ndarray]:
        """
        从图像中提取手部关键点
        
        Args:
            image_path: 图像路径
            
        Returns:
            shape (63,) 的numpy数组，包含21个关键点的x,y,z坐标
            如果未检测到手部，返回None
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测手部
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # 提取第一只手的关键点
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 转换为numpy数组
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return np.array(landmarks, dtype=np.float32)
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        归一化关键点坐标
        - 以手腕(第0个点)为原点进行平移
        - 缩放到单位范围
        
        Args:
            landmarks: shape (63,) 或 (21, 3)
            
        Returns:
            归一化后的关键点
        """
        landmarks = landmarks.reshape(21, 3)
        
        # 以手腕为原点
        wrist = landmarks[0].copy()
        landmarks = landmarks - wrist
        
        # 缩放归一化
        max_distance = np.max(np.linalg.norm(landmarks, axis=1))
        if max_distance > 0:
            landmarks = landmarks / max_distance
        
        return landmarks.reshape(-1)
    
    def close(self):
        """关闭MediaPipe"""
        self.hands.close()


def create_adjacency_matrix() -> np.ndarray:
    """
    创建手部关键点的邻接矩阵（用于GCN）
    
    返回21x21的邻接矩阵，表示关键点之间的连接关系
    """
    # 手部骨架连接关系
    # 0: 手腕
    # 1-4: 大拇指
    # 5-8: 食指
    # 9-12: 中指
    # 13-16: 无名指
    # 17-20: 小指
    
    edges = [
        # 手腕到手指根部
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # 大拇指
        (1, 2), (2, 3), (3, 4),
        # 食指
        (5, 6), (6, 7), (7, 8),
        # 中指
        (9, 10), (10, 11), (11, 12),
        # 无名指
        (13, 14), (14, 15), (15, 16),
        # 小指
        (17, 18), (18, 19), (19, 20),
    ]
    
    # 创建邻接矩阵
    num_nodes = 21
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    for i, j in edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # 无向图
    
    # 添加自环
    adj_matrix += np.eye(num_nodes)
    
    # 归一化（对称归一化）
    degree = np.sum(adj_matrix, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    degree_matrix = np.diag(degree_inv_sqrt)
    adj_matrix_normalized = degree_matrix @ adj_matrix @ degree_matrix
    
    return adj_matrix_normalized