import os
import shutil

# 定义源文件夹和目标文件夹路径
source_dir = '/dataset-b-frames-50-original1'
part_dirs = ['/dataset-b-frames-50-original-part1', '/dataset-b-frames-50-original-part2',
             '/dataset-b-frames-50-original-part3', '/dataset-b-frames-50-original-part4']

# 获取源文件夹中的所有子文件夹，并按名称排序
folders = sorted([f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))])

# 确保有124个文件夹
if len(folders) != 124:
    raise ValueError("源文件夹中不包含124个子文件夹。")

# 每部分应包含31个文件夹
num_folders_per_part = 31

# 遍历每个目标文件夹并移动相应数量的子文件夹
for i, part_dir in enumerate(part_dirs):
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)

    start_index = i * num_folders_per_part
    end_index = (i + 1) * num_folders_per_part
    for folder in folders[start_index:end_index]:
        shutil.move(os.path.join(source_dir, folder), os.path.join(part_dir, folder))

print("文件夹已成功均分并移动到目标路径下。")
