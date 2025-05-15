import os
import argparse

def clean_specific_files(root_dir):
    """清理以HL.JPG和HR.JPG结尾的文件"""
    removed_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 匹配需要删除的文件模式
            if filename.upper().endswith(('HL.JPG', 'HR.JPG')):
                file_path = os.path.join(dirpath, filename)

                # 安全删除操作
                try:
                    os.remove(file_path)
                    removed_count += 1
                    print(f"已删除：{file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {str(e)}")

    print(f"清理完成，共删除 {removed_count} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                      default='/civi/data/FER/DAN/datasets/AffectNet_KDEF/train/',
                      help='需要清理的目录路径')
    args = parser.parse_args()

    print("!!! 注意：本操作不可逆，请确认已备份数据 !!!")
    confirm = input("确认执行清理操作？(y/n): ")

    if confirm.lower() == 'y':
        clean_specific_files(args.dir)
    else:
        print("操作已取消")
