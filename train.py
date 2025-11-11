import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return np.sqrt((point1[0] - point2[0])**2 + 
                   (point1[1] - point2[1])**2 + 
                   (point1[2] - point2[2])**2)

def calculate_angle(v1, v2):
    """
    计算两个向量之间的夹角（度数）
    返回 0-180 度之间的角度
    """
    # 计算向量的模
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    # 计算夹角的余弦值
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    
    # 限制在 [-1, 1] 范围内，避免数值误差
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 计算角度（弧度转度数）
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    # 确保角度在 0-180 度之间
    if angle > 180:
        angle = 360 - angle
    
    return angle

def extract_hand_features(mediapipe_data):
    """
    从63个 MediaPipe 数据提取特征
    mediapipe_data: 63个值 (21个点 × 3个坐标)
    返回: 45 + 36 + 48 + 10 = 139 个特征
    """
    # 将一维数据转换为 21x3 的点数组
    points = mediapipe_data.reshape(21, 3)
    
    features = []
    
    # ===== 特征1: 手指段的单位方向向量 (45个特征) =====
    fingers = [
        [0, 1, 2, 3, 4],      # 拇指
        [0, 5, 6, 7, 8],      # 食指
        [0, 9, 10, 11, 12],   # 中指
        [0, 13, 14, 15, 16],  # 无名指
        [0, 17, 18, 19, 20]   # 小指
    ]
    
    for finger in fingers:
        for i in range(1, 4):  # 3段
            p1 = points[finger[i]]
            p2 = points[finger[i+1]]
            
            dist = calculate_distance(p1, p2)
            if dist > 0:
                unit_vector = (p2 - p1) / dist
                features.extend(unit_vector)
            else:
                features.extend([0, 0, 0])
    
    # ===== 特征2: 相邻手指关节差值比例 (36个特征) =====
    cmc_indices = [1, 5, 9, 13, 17]
    joint_groups = [
        [2, 6, 10, 14, 18],
        [3, 7, 11, 15, 19],
        [4, 8, 12, 16, 20]
    ]
    
    for joint_indices in joint_groups:
        for i in range(4):
            curr_diff = points[joint_indices[i+1]] - points[joint_indices[i]]
            cmc_diff = points[cmc_indices[i+1]] - points[cmc_indices[i]]
            
            for j in range(3):
                # 修改: 使用新的比例计算方法
                ratio = curr_diff[j] / (abs(cmc_diff[j]) + 0.01)
                features.append(ratio)
    
    # ===== 特征3: 相邻手指对应关节的单位方向向量 (48个特征) =====
    finger_joints = [
        [1, 2, 3, 4],      # 拇指
        [5, 6, 7, 8],      # 食指
        [9, 10, 11, 12],   # 中指
        [13, 14, 15, 16],  # 无名指
        [17, 18, 19, 20]   # 小指
    ]
    
    for joint_idx in range(4):
        for i in range(4):
            p1 = points[finger_joints[i][joint_idx]]
            p2 = points[finger_joints[i+1][joint_idx]]
            
            dist = calculate_distance(p1, p2)
            if dist > 0:
                unit_vector = (p2 - p1) / dist
                features.extend(unit_vector)
            else:
                features.extend([0, 0, 0])
    
    # ===== 特征4: 相邻指节的夹角 (10个特征) =====
    # 每根手指有2个夹角（3个关节点形成2个夹角）
    for finger in fingers:
        # 第一个夹角：关节1-2-3
        v1 = points[finger[1]] - points[finger[2]]  # 向量从关节2指向关节1
        v2 = points[finger[3]] - points[finger[2]]  # 向量从关节2指向关节3
        angle1 = calculate_angle(v1, v2)
        features.append(angle1)
        
        # 第二个夹角：关节2-3-4
        v1 = points[finger[2]] - points[finger[3]]  # 向量从关节3指向关节2
        v2 = points[finger[4]] - points[finger[3]]  # 向量从关节3指向关节4
        angle2 = calculate_angle(v1, v2)
        features.append(angle2)
    
    return features

def scan_feature_classes(feature_folder):
    """扫描特征文件夹，获取所有类别"""
    feature_folder = Path(feature_folder)
    classes = []
    
    for feature_file in feature_folder.glob("class_*_mediapipe.txt"):
        try:
            filename = feature_file.stem
            class_id = int(filename.split('_')[1])
            classes.append(class_id)
        except:
            continue
    
    return sorted(classes)

def load_features(feature_folder):
    """加载所有类别的 MediaPipe 数据并计算特征"""
    X = []  # 特征
    y = []  # 标签
    
    feature_folder = Path(feature_folder)
    
    # 自动扫描所有类别
    all_classes = scan_feature_classes(feature_folder)
    
    if not all_classes:
        print("错误: 未找到任何 MediaPipe 数据文件!")
        return None, None, None
    
    print(f"发现 {len(all_classes)} 个类别: {all_classes}")
    
    # 创建类别到索引的映射（0-based）
    class_to_idx = {class_id: idx for idx, class_id in enumerate(all_classes)}
    
    for class_id in all_classes:
        feature_file = feature_folder / f"class_{class_id}_mediapipe.txt"
        
        if not feature_file.exists():
            print(f"警告: 找不到文件 {feature_file}")
            continue
        
        # 读取 MediaPipe 数据
        count = 0
        with open(feature_file, 'r') as f:
            for line in f:
                mediapipe_data = np.array([float(x) for x in line.strip().split()])
                if len(mediapipe_data) == 63:  # 确保数据完整
                    # 计算特征
                    features = extract_hand_features(mediapipe_data)
                    X.append(features)
                    y.append(class_to_idx[class_id])
                    count += 1
        
        print(f"类别 {class_id} (索引 {class_to_idx[class_id]}): 加载了 {count} 个样本")
    
    return np.array(X), np.array(y), class_to_idx

def train_classifier(feature_folder, model_save_path="gesture_model.pkl", 
                    use_scaler=False, boosting_type='dart'):
    """
    训练 LightGBM 分类器
    
    参数:
        feature_folder: 特征文件夹路径
        model_save_path: 模型保存路径
        use_scaler: 是否使用特征归一化 (默认False)
        boosting_type: 提升类型 'dart' 或 'gbdt' (默认'dart')
    """
    print("正在加载 MediaPipe 数据并计算特征...")
    X, y, class_to_idx = load_features(feature_folder)
    
    if X is None or len(X) == 0:
        print("错误: 没有加载到任何数据!")
        return None
    
    num_classes = len(class_to_idx)
    idx_to_class = {idx: class_id for class_id, idx in class_to_idx.items()}
    
    print(f"\n总样本数: {len(X)}")
    print(f"特征维度: {X.shape[1]} (45 + 36 + 48 + 10 = 139)")
    print(f"类别数量: {num_classes}")
    print(f"标签范围: {y.min()} - {y.max()}")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集样本数: {len(X_train)}")
    print(f"测试集样本数: {len(X_test)}")
    
    # 特征归一化 (可选)
    scaler = None
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        print("\n使用 StandardScaler 进行特征归一化...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("✓ 特征归一化完成")
    else:
        print("\n未使用特征归一化")
    
    # 创建 LightGBM 数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 设置参数
    print(f"\n使用 boosting 类型: {boosting_type}")
    
    if boosting_type == 'dart':
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'dart',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'drop_rate': 0.2,
            'max_drop': 50,
            'skip_drop': 0.5,
            'xgboost_dart_mode': False,
            'uniform_drop': False
        }
    
    print(f"\n开始训练 LightGBM ({boosting_type.upper()}) 模型 ({num_classes} 个类别)...")
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # 预测
    print("\n评估模型...")
    y_pred_train = np.argmax(model.predict(X_train), axis=1)
    y_pred_test = np.argmax(model.predict(X_test), axis=1)
    
    # 计算准确率
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 详细分类报告
    print("\n分类报告:")
    target_names = [f"Class {idx_to_class[i]}" for i in range(num_classes)]
    print(classification_report(y_test, y_pred_test, 
                                target_names=target_names,
                                labels=list(range(num_classes))))
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred_test, labels=list(range(num_classes)))
    print(cm)
    
    # 各类别准确率
    print("\n各类别准确率:")
    for i in range(num_classes):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
            print(f"Class {idx_to_class[i]}: {acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    # ========== 诊断检查 1: 预测概率分析 ==========
    print("\n" + "="*60)
    print("【诊断 1】各类别预测概率分析")
    print("="*60)
    y_pred_probs_test = model.predict(X_test)
    
    problem_classes = []
    for i in range(num_classes):
        class_mask = (y_test == i)
        if class_mask.sum() > 0:
            # 该类别样本的预测概率
            class_probs = y_pred_probs_test[class_mask][:, i]
            mean_prob = class_probs.mean()
            max_prob = class_probs.max()
            min_prob = class_probs.min()
            
            print(f"\n类别 {idx_to_class[i]}:")
            print(f"  测试集样本数: {class_mask.sum()}")
            print(f"  预测为自己的平均概率: {mean_prob:.4f}")
            print(f"  预测为自己的最大概率: {max_prob:.4f}")
            print(f"  预测为自己的最小概率: {min_prob:.4f}")
            
            # 找出该类别最常被预测成什么
            max_prob_class = np.argmax(y_pred_probs_test[class_mask].mean(axis=0))
            print(f"  最常被预测为: 类别 {idx_to_class[max_prob_class]}")
            
            # 检测问题类别
            if max_prob < 0.3:  # 如果最大概率都小于0.3
                problem_classes.append(idx_to_class[i])
                print(f"  ⚠️ 警告: 该类别预测概率异常低!")
    
    if problem_classes:
        print(f"\n⚠️ 发现 {len(problem_classes)} 个问题类别: {problem_classes}")
    else:
        print("\n✓ 所有类别预测概率正常")
    
    # ========== 诊断检查 2: 特征范围对比 ==========
    print("\n" + "="*60)
    print("【诊断 2】各类别特征范围对比")
    print("="*60)
    
    # 重建 all_classes 列表
    all_classes_sorted = sorted(class_to_idx.keys())
    
    print("\n检查每组特征的第一个维度:")
    feature_groups = {
        "特征1 (方向向量)": 0,
        "特征2 (比例)": 45,
        "特征3 (方向向量)": 81,
        "特征4 (角度)": 129
    }
    
    for group_name, feature_idx in feature_groups.items():
        print(f"\n{group_name} - 特征索引 {feature_idx}:")
        ranges = []
        for class_id in all_classes_sorted[:min(5, len(all_classes_sorted))]:
            class_mask = (y == class_to_idx[class_id])
            class_features = X[class_mask, feature_idx]
            min_val = class_features.min()
            max_val = class_features.max()
            ranges.append(max_val - min_val)
            print(f"  类别 {class_id}: [{min_val:.3f}, {max_val:.3f}]")
        
        # 检查范围差异
        if len(ranges) > 0 and max(ranges) / (min(ranges) + 1e-6) > 100:
            print(f"  ⚠️ 警告: 类别间特征范围差异过大! (建议使用特征归一化)")
    
    # ========== 诊断检查 3: 角度特征异常检测 ==========
    print("\n" + "="*60)
    print("【诊断 3】角度特征异常检测")
    print("="*60)
    
    if X.shape[1] >= 139:
        angle_features = X[:, 129:139]  # 最后10个特征是角度
        
        angle_issues = []
        for i in range(10):
            angle_col = angle_features[:, i]
            min_angle = angle_col.min()
            max_angle = angle_col.max()
            mean_angle = angle_col.mean()
            
            print(f"\n角度特征 {i} (手指 {i//2}, 角度 {i%2}):")
            print(f"  范围: [{min_angle:.1f}°, {max_angle:.1f}°]")
            print(f"  平均: {mean_angle:.1f}°")
            
            # 检查是否有异常值（角度应该在0-180之间）
            invalid_count = ((angle_col < 0) | (angle_col > 180)).sum()
            if invalid_count > 0:
                print(f"  ⚠️ 发现 {invalid_count} 个异常角度值 (不在0-180范围)!")
                angle_issues.append(f"角度{i}有{invalid_count}个异常值")
            
            # 检查是否过多为0（说明向量长度为0）
            zero_count = (angle_col == 0).sum()
            zero_ratio = zero_count / len(angle_col)
            if zero_ratio > 0.5:
                print(f"  ⚠️ {zero_count}/{len(angle_col)} ({zero_ratio:.1%}) 个样本角度为0")
                angle_issues.append(f"角度{i}有{zero_ratio:.1%}为0")
        
        if angle_issues:
            print(f"\n⚠️ 角度特征存在问题: {len(angle_issues)} 个")
        else:
            print("\n✓ 所有角度特征正常")
    
    # ========== 诊断检查 4: 类别映射检查 ==========
    print("\n" + "="*60)
    print("【诊断 4】类别映射检查")
    print("="*60)
    
    print("\nclass_to_idx:", class_to_idx)
    print("idx_to_class:", idx_to_class)
    
    # 检查映射完整性
    assert len(class_to_idx) == len(idx_to_class), "映射长度不一致!"
    assert set(class_to_idx.values()) == set(range(num_classes)), "索引不连续!"
    print("\n✓ 类别映射正确")
    
    # ========== 诊断检查 5: 零准确率类别检测 ==========
    print("\n" + "="*60)
    print("【诊断 5】零准确率类别检测")
    print("="*60)
    
    zero_acc_classes = []
    for i in range(num_classes):
        if cm[i].sum() > 0 and cm[i, i] == 0:
            zero_acc_classes.append(idx_to_class[i])
            print(f"\n⚠️ 类别 {idx_to_class[i]}: 测试集准确率为 0")
            print(f"   该类别的 {cm[i].sum()} 个样本被误判为:")
            for j in range(num_classes):
                if cm[i, j] > 0:
                    print(f"     类别 {idx_to_class[j]}: {cm[i, j]} 次")
    
    if zero_acc_classes:
        print(f"\n⚠️ 发现 {len(zero_acc_classes)} 个零准确率类别: {zero_acc_classes}")
    else:
        print("\n✓ 所有类别在测试集上都有正确预测")
    
    # ========== 诊断检查 6: 样本数量统计 ==========
    print("\n" + "="*60)
    print("【诊断 6】样本数量分析")
    print("="*60)
    
    sample_counts = {}
    for class_id in class_to_idx.keys():
        class_mask = (y == class_to_idx[class_id])
        sample_counts[class_id] = class_mask.sum()
    
    min_samples = min(sample_counts.values())
    max_samples = max(sample_counts.values())
    min_class = [k for k, v in sample_counts.items() if v == min_samples][0]
    max_class = [k for k, v in sample_counts.items() if v == max_samples][0]
    
    print(f"\n样本数量范围: {min_samples} - {max_samples}")
    print(f"样本最少的类别: {min_class} ({min_samples} 个)")
    print(f"样本最多的类别: {max_class} ({max_samples} 个)")
    print(f"样本数量比: {max_samples}/{min_samples} = {max_samples/min_samples:.2f}")
    
    if max_samples / min_samples > 5:
        print(f"\n⚠️ 警告: 类别不平衡严重! 建议:")
        print("  1. 为样本少的类别增加更多训练数据")
        print("  2. 使用类别权重平衡")
        print("  3. 使用过采样/欠采样技术")
    
    # ========== 诊断总结 ==========
    print("\n" + "="*60)
    print("【诊断总结】")
    print("="*60)
    
    issues = []
    if problem_classes:
        issues.append(f"预测概率异常类别: {problem_classes}")
    if zero_acc_classes:
        issues.append(f"零准确率类别: {zero_acc_classes}")
    if max_samples / min_samples > 5:
        issues.append(f"类别不平衡严重 (比例 {max_samples/min_samples:.1f}:1)")
    
    if issues:
        print("\n发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n建议:")
        print("  - 使用特征归一化 (StandardScaler)")
        print("  - 尝试切换到 gbdt 而不是 dart")
        print("  - 增加问题类别的训练样本")
        print("  - 检查问题类别的图片质量")
    else:
        print("\n✓ 未发现明显问题")
    
    # 保存模型和映射
    print("\n" + "="*60)
    print(f"保存模型到 {model_save_path}...")
    print("="*60)
    model_data = {
        'model': model,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'scaler': scaler  # 保存归一化器（如果使用了）
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\n训练完成!")
    return model

if __name__ == "__main__":
    import sys
    
    feature_folder = "gesture_3"
    model_save_path = "gesture_model.pkl"
    
    # 解析命令行参数
    use_scaler = False
    boosting_type = 'dart'
    
    if len(sys.argv) > 1:
        if '--scaler' in sys.argv or '-s' in sys.argv:
            use_scaler = True

    print("="*60)
    print("训练配置")
    print("="*60)
    print(f"特征文件夹: {feature_folder}")
    print(f"模型保存路径: {model_save_path}")
    print(f"特征归一化: {'启用' if use_scaler else '禁用'}")
    print(f"Boosting类型: {boosting_type.upper()}")
    print("\n使用方法:")
    print("  python train.py              # 默认配置")
    print("  python train.py --scaler     # 启用特征归一化")
    print("  python train.py --gbdt       # 使用GBDT而不是DART")
    print("  python train.py -s -g        # 同时启用两者")
    print("="*60 + "\n")
    
    if not Path(feature_folder).exists():
        print(f"错误: 找不到特征文件夹 '{feature_folder}'")
    else:
        train_classifier(feature_folder, model_save_path, use_scaler, boosting_type)