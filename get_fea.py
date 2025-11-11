import cv2
import mediapipe as mp
import numpy as np
import os
from pathlib import Path
from collections import defaultdict

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_mediapipe_data(landmarks):
    """
    提取 MediaPipe 的 63 个原始坐标值 (21个点 × 3个坐标)
    landmarks: MediaPipe返回的21个关节点 (x, y, z)
    """
    # 将landmarks转换为numpy数组
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # 展平为一维数组 (63个值)
    mediapipe_data = points.flatten()
    
    return mediapipe_data

def get_class_from_filename(filename):
    """从文件名中提取类别ID"""
    try:
        # 去掉扩展名
        name = filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', '')
        # 分割获取第一个数字
        class_id = int(name.split('_')[0])
        return class_id
    except:
        return None

def scan_classes(input_folder):
    """扫描文件夹，获取所有类别ID"""
    classes = set()
    image_files = list(Path(input_folder).glob('*.jpg')) + \
                  list(Path(input_folder).glob('*.png')) + \
                  list(Path(input_folder).glob('*.jpeg'))
    
    for img_path in image_files:
        class_id = get_class_from_filename(img_path.name)
        if class_id is not None:
            classes.add(class_id)
    
    return sorted(classes)

def process_images(input_folder, output_folder):
    """处理所有图片并生成 MediaPipe 数据文件"""
    # 创建输出文件夹
    Path(output_folder).mkdir(exist_ok=True)
    
    # 首先扫描所有类别
    print("正在扫描类别...")
    all_classes = scan_classes(input_folder)
    
    if not all_classes:
        print("错误: 未找到有效的类别!")
        return
    
    print(f"发现 {len(all_classes)} 个类别: {all_classes}")
    print(f"类别范围: {min(all_classes)} - {max(all_classes)}")
    
    # 为每个类别创建字典存储数据
    class_data = defaultdict(list)
    
    # 统计信息
    total_images = 0
    processed_images = 0
    failed_images = 0
    class_counts = defaultdict(int)
    
    # 获取所有图片文件
    image_files = list(Path(input_folder).glob('*.jpg')) + \
                  list(Path(input_folder).glob('*.png')) + \
                  list(Path(input_folder).glob('*.jpeg'))
    total_images = len(image_files)
    
    print(f"\n找到 {total_images} 张图片")
    print("开始处理...\n")
    
    for img_path in image_files:
        filename = img_path.name
        
        try:
            # 解析文件名获取类别
            class_id = get_class_from_filename(filename)
            
            if class_id is None:
                print(f"跳过无效文件名: {filename}")
                failed_images += 1
                continue
            
            if class_id not in all_classes:
                print(f"跳过未知类别 {class_id}: {filename}")
                failed_images += 1
                continue
            
            # 读取图片
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"无法读取图片: {filename}")
                failed_images += 1
                continue
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用MediaPipe检测手部
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # 提取 MediaPipe 原始数据
                landmarks = results.multi_hand_landmarks[0].landmark
                mediapipe_data = extract_mediapipe_data(landmarks)
                
                # 存储数据
                class_data[class_id].append(mediapipe_data)
                class_counts[class_id] += 1
                processed_images += 1
                
                if processed_images % 100 == 0:
                    print(f"已处理: {processed_images}/{total_images}")
            else:
                print(f"未检测到手部: {filename}")
                failed_images += 1
                
        except Exception as e:
            print(f"处理图片 {filename} 时出错: {e}")
            failed_images += 1
    
    # 保存数据到txt文件
    print("\n正在保存 MediaPipe 数据文件...")
    for class_id in sorted(class_data.keys()):
        if len(class_data[class_id]) > 0:
            output_file = Path(output_folder) / f"class_{class_id}_mediapipe.txt"
            
            with open(output_file, 'w') as f:
                for data in class_data[class_id]:
                    # 将数据转换为字符串，用空格分隔
                    data_str = ' '.join([f"{val:.6f}" for val in data])
                    f.write(data_str + '\n')
            
            print(f"类别 {class_id}: 保存了 {len(class_data[class_id])} 个样本")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"{'='*60}")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {processed_images}")
    print(f"失败/跳过: {failed_images}")
    print(f"检测到的类别数: {len(all_classes)}")
    print(f"MediaPipe 数据维度: 63 (21个点 × 3个坐标)")
    print(f"\n各类别样本数:")
    for class_id in sorted(class_counts.keys()):
        print(f"  类别 {class_id}: {class_counts[class_id]} 个样本")

if __name__ == "__main__":
    # 设置输入输出路径
    input_folder = "c2"
    output_folder = "gesture_1110"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 找不到文件夹 '{input_folder}'")
        print("请确保 camera_shots 文件夹存在")
    else:
        process_images(input_folder, output_folder)
    
    # 关闭MediaPipe
    hands.close()
