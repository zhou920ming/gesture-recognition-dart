import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path
import time

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    if angle > 180:
        angle = 360 - angle
    
    return angle

def extract_hand_features(landmarks):
    """
    从21个关节点提取特征
    返回: 45 + 36 + 48 + 10 = 139 个特征
    """
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    features = []
    
    # 特征1: 手指段的单位方向向量 (45个特征)
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    
    for finger in fingers:
        for i in range(1, 4):
            p1 = points[finger[i]]
            p2 = points[finger[i+1]]
            dist = calculate_distance(p1, p2)
            if dist > 0:
                unit_vector = (p2 - p1) / dist
                features.extend(unit_vector)
            else:
                features.extend([0, 0, 0])
    
    # 特征2: 相邻手指关节差值比例 (36个特征)
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
    
    # 特征3: 相邻手指对应关节的单位方向向量 (48个特征)
    finger_joints = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
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
    
    # 特征4: 相邻指节的夹角 (10个特征)
    for finger in fingers:
        # 第一个夹角：关节1-2-3
        v1 = points[finger[1]] - points[finger[2]]
        v2 = points[finger[3]] - points[finger[2]]
        angle1 = calculate_angle(v1, v2)
        features.append(angle1)
        
        # 第二个夹角：关节2-3-4
        v1 = points[finger[2]] - points[finger[3]]
        v2 = points[finger[4]] - points[finger[3]]
        angle2 = calculate_angle(v1, v2)
        features.append(angle2)
    
    return np.array(features)

def load_model(model_path):
    """加载训练好的模型"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def real_time_gesture_recognition(model_path="gesture_model.pkl"):
    """实时手势识别"""
    if not Path(model_path).exists():
        print(f"错误: 找不到模型文件 '{model_path}'")
        print("请先运行训练程序生成模型")
        return
    
    # 加载模型
    print("正在加载模型...")
    model_data = load_model(model_path)
    model = model_data['model']
    idx_to_class = model_data['idx_to_class']
    num_classes = len(idx_to_class)
    
    print(f"模型加载成功! 类别数: {num_classes}")
    print(f"类别列表: {sorted(idx_to_class.values())}")
    print(f"特征维度: 139 (45 + 36 + 48 + 10)")
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 初始化 MediaPipe Hands
    hands = mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    )
    
    print("\n摄像头已启动!")
    print("按 'q' 键退出")
    print("=" * 50)
    
    # 用于平滑预测结果
    prediction_history = []
    history_size = 5
    
    # FPS 计算
    prev_time = time.time()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法读取摄像头")
            break
        
        # 翻转图像
        image = cv2.flip(image, 1)
        
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # 检测手部
        results = hands.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 计算 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # 如果检测到手部
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # 提取特征
                features = extract_hand_features(hand_landmarks.landmark)
                features = features.reshape(1, -1)
                
                # 预测
                prediction_probs = model.predict(features)[0]
                predicted_idx = np.argmax(prediction_probs)
                predicted_class = idx_to_class[predicted_idx]
                confidence = prediction_probs[predicted_idx]
                
                # 添加到历史记录
                prediction_history.append(predicted_class)
                if len(prediction_history) > history_size:
                    prediction_history.pop(0)
                
                # 使用投票方式平滑预测结果
                if len(prediction_history) >= 3:
                    final_prediction = max(set(prediction_history), 
                                          key=prediction_history.count)
                else:
                    final_prediction = predicted_class
                
                # 显示预测结果
                text = f"Gesture: Class {final_prediction}"
                conf_text = f"Confidence: {confidence:.2f}"
                
                # 在图像上显示文字
                cv2.putText(image, text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
                cv2.putText(image, conf_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                # 显示 FPS
                cv2.putText(image, f"FPS: {fps:.1f}", (image.shape[1] - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # 显示前10个类别的概率
                y_offset = 150
                top_k = min(10, num_classes)
                top_indices = np.argsort(prediction_probs)[-top_k:][::-1]
                
                for idx in top_indices:
                    class_id = idx_to_class[idx]
                    prob_text = f"Class {class_id}: {prediction_probs[idx]:.3f}"
                    
                    if class_id == final_prediction:
                        color = (0, 255, 0)
                        thickness = 2
                    else:
                        color = (200, 200, 200)
                        thickness = 1
                    
                    cv2.putText(image, prob_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
                    y_offset += 30
        else:
            # 没有检测到手部
            cv2.putText(image, "No hand detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(image, f"FPS: {fps:.1f}", (image.shape[1] - 150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            prediction_history.clear()
        
        # 显示提示信息
        cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示图像
        cv2.imshow('Gesture Recognition', image)
        
        # 按 'q' 键退出
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # 释放资源
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")

if __name__ == "__main__":
    model_path = "gesture_model.pkl"
    real_time_gesture_recognition(model_path)
