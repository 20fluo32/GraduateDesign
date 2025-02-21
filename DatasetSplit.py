import os
import random
import shutil

# 设置文件夹路径
source_folder = 'path_to_your_folder'  # 替换为你的文件夹路径
train_folder = 'path_to_train_folder'  # 替换为训练集存放的文件夹路径
test_folder = 'path_to_test_folder'  # 替换为测试集存放的文件夹路径

# 创建训练集和测试集文件夹（如果不存在）
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 获取所有.jpg文件
jpg_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# 打乱文件顺序
random.shuffle(jpg_files)

# 计算划分的索引
split_index = int(len(jpg_files) * 0.7)

# 将文件复制到训练集和测试集文件夹
for i, file in enumerate(jpg_files):
    src_file = os.path.join(source_folder, file)
    if i < split_index:
        dest_file = os.path.join(train_folder, file)
    else:
        dest_file = os.path.join(test_folder, file)

    shutil.copy(src_file, dest_file)

print(f"共 {len(jpg_files)} 个文件，已按 7:3 划分到 {train_folder} 和 {test_folder}。")
