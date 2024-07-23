import os
import shutil

# 定义源文件夹和目标文件夹路径
source_dir = '/dataset-b-frames-50-original1'
part_dirs = ['/dataset-b-frames-50-original1/part1', '/dataset-b-frames-50-original1/part2',
             '/dataset-b-frames-50-original1/part3', '/dataset-b-frames-50-original1/part4']

# 获取源文件夹中的所有子文件夹
folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

# 确保有124个文件夹
if len(folders) != 124:
    raise ValueError("源文件夹中的子文件夹数量不是124")

# 将文件夹均分成4部分，每部分31个文件夹
parts = [folders[i:i + 31] for i in range(0, len(folders), 31)]

# 确保生成的parts有4部分
if len(parts) != 4:
    raise ValueError("文件夹分组出现问题")

# 创建目标文件夹并移动文件夹
for i, part in enumerate(parts):
    part_dir = part_dirs[i]
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)
    for folder in part:
        src_path = os.path.join(source_dir, folder)
        dst_path = os.path.join(part_dir, folder)
        shutil.move(src_path, dst_path)

print("文件夹已成功移动到对应的部分路径下")
