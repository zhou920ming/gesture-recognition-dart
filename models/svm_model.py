"""
SVM Gesture Classifier
基于支持向量机算法的手势分类器
"""

import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import Optional, Tuple, Dict


class SVMGestureClassifier:
    """
    SVM手势分类器
    使用支持向量机算法进行手势识别
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 probability: bool = True, normalize: bool = True):
        """
        初始化SVM分类器
        
        Args:
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            C: 正则化参数
            gamma: 核函数系数
            probability: 是否启用概率估计
            normalize: 是否对特征进行标准化
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.normalize = normalize
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=42
        )
        
        self.scaler = StandardScaler() if normalize else None
        self.is_trained = False
        self.classes_ = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        训练SVM模型
        
        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
            
        Returns:
            包含训练信息的字典
        """
        # 标准化特征
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
        
        # 训练模型
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # 计算训练准确率
        train_accuracy = self.model.score(X_train, y_train)
        
        return {
            'train_accuracy': train_accuracy,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'n_classes': len(self.classes_),
            'n_support_vectors': self.model.n_support_.sum()
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测手势类别
        
        Args:
            X: 输入特征 (n_samples, n_features)
            
        Returns:
            预测的类别标签
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        if self.normalize:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测各类别的概率
        
        Args:
            X: 输入特征 (n_samples, n_features)
            
        Returns:
            各类别的概率 (n_samples, n_classes)
        """
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        if not self.probability:
            raise ValueError("模型未启用概率估计，请在初始化时设置probability=True")
        
        if self.normalize:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def predict_single(self, features: np.ndarray) -> Tuple[int, float]:
        """
        预测单个样本
        
        Args:
            features: 单个样本的特征向量
            
        Returns:
            (预测类别, 置信度)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        prediction = self.predict(features)[0]
        
        if self.probability:
            probabilities = self.predict_proba(features)[0]
            confidence = np.max(probabilities)
        else:
            # 使用决策函数值作为置信度
            decision_values = self.model.decision_function(
                self.scaler.transform(features) if self.normalize else features
            )
            confidence = np.max(np.abs(decision_values))
        
        return prediction, confidence
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            包含评估指标的字典
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train()方法")
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                    param_grid: Optional[Dict] = None, cv: int = 5) -> Dict:
        """
        网格搜索最优超参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数
            
        Returns:
            最优参数和得分
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        
        # 标准化
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
        
        # 网格搜索
        grid_search = GridSearchCV(
            SVC(probability=self.probability, random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 更新模型参数
        self.kernel = grid_search.best_params_['kernel']
        self.C = grid_search.best_params_['C']
        self.gamma = grid_search.best_params_['gamma']
        
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型未训练，无法保存")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'probability': self.probability,
            'normalize': self.normalize,
            'classes_': self.classes_
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.kernel = model_data['kernel']
        self.C = model_data['C']
        self.gamma = model_data['gamma']
        self.probability = model_data['probability']
        self.normalize = model_data['normalize']
        self.classes_ = model_data['classes_']
        self.is_trained = True
        
        print(f"模型已从 {filepath} 加载")
    
    def get_params(self) -> Dict:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        params = {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'probability': self.probability,
            'normalize': self.normalize,
            'is_trained': self.is_trained,
            'n_classes': len(self.classes_) if self.classes_ is not None else 0
        }
        
        if self.is_trained:
            params['n_support_vectors'] = self.model.n_support_.sum()
        
        return params
    
    def get_support_vectors(self) -> Optional[np.ndarray]:
        """
        获取支持向量
        
        Returns:
            支持向量数组
        """
        if not self.is_trained:
            return None
        return self.model.support_vectors_


if __name__ == "__main__":
    # 示例使用
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=5, 
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建和训练模型
    classifier = SVMGestureClassifier(kernel='rbf', C=1.0, normalize=True)
    train_info = classifier.train(X_train, y_train)
    print(f"训练信息: {train_info}")
    
    # 评估模型
    metrics = classifier.evaluate(X_test, y_test)
    print(f"测试指标: {metrics}")
    
    # 单个预测
    sample = X_test[0]
    pred, conf = classifier.predict_single(sample)
    print(f"预测类别: {pred}, 置信度: {conf:.4f}")
    
    # 模型参数
    params = classifier.get_params()
    print(f"模型参数: {params}")
