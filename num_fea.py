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

def get_class_from_foldername(foldername):
    """从文件夹名中提取类别ID（仅数字文件夹）"""
    try:
        class_id = int(foldername)
        return class_id
    except:
        return None

def scan_classes(input_folder):
    """扫描文件夹，获取所有数字命名的子文件夹作为类别ID"""
    classes = set()
    input_path = Path(input_folder)
    
    # 遍历所有子文件夹
    for subfolder in input_path.iterdir():
        if subfolder.is_dir():
            class_id = get_class_from_foldername(subfolder.name)
            if class_id is not None:
                classes.add(class_id)
    
    return sorted(classes)

def process_images(input_folder, output_folder):
    """处理所有图片并生成 MediaPipe 数据文件"""
    # 创建输出文件夹
    Path(output_folder).mkdir(exist_ok=True)
    
    # 首先扫描所有类别
    print("正在扫描类别文件夹...")
    all_classes = scan_classes(input_folder)
    
    if not all_classes:
        print("错误: 未找到有效的数字命名文件夹!")
        return
    
    print(f"发现 {len(all_classes)} 个类别: {all_classes}")
    print(f"类别范围: {min(all_classes)} - {max(all_classes)}")
    print()
    
    # 统计信息
    total_images = 0
    processed_images = 0
    failed_images = 0
    class_counts = defaultdict(int)
    
    # 逐个文件夹处理
    for class_id in all_classes:
        class_folder = Path(input_folder) / str(class_id)
        
        print(f"{'='*60}")
        print(f"处理类别 {class_id}: {class_folder}")
        print(f"{'='*60}")
        
        # 获取该类别文件夹下的所有图片
        image_files = list(class_folder.glob('*.jpg')) + \
                      list(class_folder.glob('*.png')) + \
                      list(class_folder.glob('*.jpeg'))
        
        folder_total = len(image_files)
        folder_processed = 0
        folder_failed = 0
        
        print(f"找到 {folder_total} 张图片")
        
        # 存储该类别的数据
        class_data = []
        
        for img_path in image_files:
            filename = img_path.name
            
            try:
                # 读取图片
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"  无法读取图片: {filename}")
                    folder_failed += 1
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
                    class_data.append(mediapipe_data)
                    folder_processed += 1
                    
                    if folder_processed % 50 == 0:
                        print(f"  已处理: {folder_processed}/{folder_total}")
                else:
                    # print(f"  未检测到手部: {filename}")
                    folder_failed += 1
                    
            except Exception as e:
                print(f"  处理图片 {filename} 时出错: {e}")
                folder_failed += 1
        
        # 保存该类别的数据到txt文件
        if len(class_data) > 0:
            output_file = Path(output_folder) / f"class_{class_id}_mediapipe.txt"
            
            with open(output_file, 'w') as f:
                for data in class_data:
                    # 将数据转换为字符串，用空格分隔
                    data_str = ' '.join([f"{val:.6f}" for val in data])
                    f.write(data_str + '\n')
            
            print(f"\n类别 {class_id} 处理完成:")
            print(f"  成功: {folder_processed} 个样本")
            print(f"  失败: {folder_failed} 个样本")
            print(f"  保存到: {output_file}")
        else:
            print(f"\n警告: 类别 {class_id} 没有有效数据!")
        
        # 更新总统计
        total_images += folder_total
        processed_images += folder_processed
        failed_images += folder_failed
        class_counts[class_id] = folder_processed
        
        print()
    
    # 打印总体统计信息
    print(f"\n{'='*60}")
    print(f"全部处理完成!")
    print(f"{'='*60}")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {processed_images}")
    print(f"失败/跳过: {failed_images}")
    print(f"处理的类别数: {len(all_classes)}")
    print(f"MediaPipe 数据维度: 63 (21个点 × 3个坐标)")
    print(f"\n各类别样本数:")
    for class_id in sorted(class_counts.keys()):
        print(f"  类别 {class_id}: {class_counts[class_id]} 个样本")

if __name__ == "__main__":
    # 设置输入输出路径
    # input_folder 应该包含多个数字命名的子文件夹，如: 1/, 2/, 3/, ...
    # 每个子文件夹包含该类别的所有图片
    input_folder = "tem"  # 改为你的主文件夹路径
    output_folder = "tem"
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"错误: 找不到文件夹 '{input_folder}'")
        print("请确保文件夹存在，且包含数字命名的子文件夹")
        print("示例结构:")
        print("  gesture_data/")
        print("    1/")
        print("      img001.jpg")
        print("      img002.jpg")
        print("    2/")
        print("      img001.jpg")
        print("      img002.jpg")
    else:
        process_images(input_folder, output_folder)
    
    # 关闭MediaPipe
    hands.close()