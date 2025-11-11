def merge_txt_files(source_file, target_file):
    """
    将source_file的内容添加到target_file中
    
    参数:
        source_file: 源文件名(要读取的文件)
        target_file: 目标文件名(要追加内容的文件)
    """
    try:
        # 读取源文件内容
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 将内容追加到目标文件
        with open(target_file, 'a', encoding='utf-8') as f:
            f.write(content)
        
        print(f"成功将 '{source_file}' 的内容添加到 '{target_file}'")
        
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except Exception as e:
        print(f"发生错误: {e}")


# 主程序
if __name__ == "__main__":
    # 输入文件名
    source = input("请输入源文件名(包含.txt后缀): ")
    target = input("请输入目标文件名(包含.txt后缀): ")
    
    # 执行合并
    merge_txt_files(source, target)
