import os
import shutil

# 定义源文件夹和目标文件夹路径
source_dir = 'E:/CASIA_Gait_Dataset/DatasetB'
target_dir1 = 'E:/CASIA_Gait_Dataset/DatasetB-train'
target_dir2 = 'E:/CASIA_Gait_Dataset/DatasetB-test'

# 获取源文件夹中的所有子文件夹，并按名称排序
folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])

# 确保有124个文件夹
if len(folders) != 124:
    raise ValueError("源文件夹中不包含124个子文件夹。")

# 前74个文件夹移动到目标文件夹1
if not os.path.exists(target_dir1):
    os.makedirs(target_dir1)

for folder in folders[:74]:
    shutil.move(os.path.join(source_dir, folder), os.path.join(target_dir1, folder))

# 后50个文件夹移动到目标文件夹2
if not os.path.exists(target_dir2):
    os.makedirs(target_dir2)

for folder in folders[74:]:
    shutil.move(os.path.join(source_dir, folder), os.path.join(target_dir2, folder))

print("文件夹已成功划分并移动到目标路径下。")
