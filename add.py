import os
import re
from pathlib import Path

def append_features_files(source_folder, target_folder):
    """
    读取source_folder下所有class_n_features.txt文件，
    追加到target_folder下的对应同名文件中
    
    参数:
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)
    
    # 匹配文件名模式: class_数字_features.txt
    pattern = re.compile(r'^class_\d+_mediapipe\.txt$')
    
    # 遍历源文件夹
    for filename in os.listdir(source_folder):
        if pattern.match(filename):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)
            
            # 读取源文件内容
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 追加到目标文件
            with open(target_path, 'a', encoding='utf-8') as f:
                f.write(content)
            
            print(f"已追加: {filename}")
    
    print("所有文件处理完成!")

# 使用示例
if __name__ == "__main__":
    source_folder = "tem"  # 替换为你的源文件夹路径
    target_folder = "gesture_3"  # 替换为你的目标文件夹路径
    
    append_features_files(source_folder, target_folder)