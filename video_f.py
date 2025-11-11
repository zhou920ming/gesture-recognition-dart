import cv2
import os
from pathlib import Path

def sample_video_frames(video_path, output_folder, video_index, target_frames=300):
    """
    从视频中均匀采样指定数量的帧
    
    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹路径
        video_index: 视频索引(1-26)
        target_frames: 目标采样帧数，默认300
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"视频帧数为0: {video_path}")
        cap.release()
        return
    
    # 计算采样间隔
    interval = max(1, total_frames // target_frames)
    
    frame_count = 0
    saved_count = 0
    
    print(f"处理视频 {video_index}: {os.path.basename(video_path)}")
    print(f"  总帧数: {total_frames}, 采样间隔: {interval}")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 每隔interval帧保存一张图片
        if frame_count % interval == 0:
            output_path = os.path.join(output_folder, f"{video_index}_{saved_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"  已保存 {saved_count} 张图片\n")

def main():
    # 设置路径
    video_folder = "g2"
    output_folder = "c2"
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有MP4文件并排序
    video_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')])
    
    if len(video_files) == 0:
        print(f"在 {video_folder} 文件夹中没有找到MP4文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件\n")
    
    # 处理每个视频
    for idx, video_file in enumerate(video_files, start=1):
        video_path = os.path.join(video_folder, video_file)
        sample_video_frames(video_path, output_folder, idx, target_frames=300)
    
    print("所有视频处理完成！")

if __name__ == "__main__":
    main()