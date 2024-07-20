import argparse
import os
import glob
import shutil

# 创建一个ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser(description='Test Images')
# 添加一个参数，用于指定视频文件的目录，并且该参数是必需的
parser.add_argument('--videos-dir', type=str, required=True)
# 添加一个参数，用于指定结果文件的目录，并且该参数是必需的
parser.add_argument('--result-dir', type=str, required=True)

# 解析命令行参数
args = parser.parse_args()

# 加载视频文件
# 使用glob模块找到所有指定目录及其子目录下的所有avi文件
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.png'), recursive=True)])

# 获取视频文件的数量
num_video = len(video_list)
print("Found", num_video, "videos")  # 打印找到的视频文件数量

# 处理每个视频文件
for i in range(num_video):
    video_path = video_list[i]  # 获取当前视频文件的路径
    video_name = os.path.basename(video_path)  # 获取文件名（包含扩展名）
    image_name, _ = os.path.splitext(video_name)  # 去除扩展名
    print(i, '/', num_video, image_name)  # 打印当前处理的文件索引及文件名

    # 解析文件名以创建目录结构
    parts = image_name.split('-')  # 按照 '-' 分割文件名
    print(f"Filename parts: {parts}")  # 打印文件名的各部分
    if len(parts) == 5:  # 如果分割后的部分数量为5，说明文件名格式正确
        name = parts[4]
    else:  # 如果文件名格式不符合预期，跳过该文件
        print(f"Unexpected filename format: {image_name}")
        continue

    # 保存结果
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 生成保存路径
    save_path = os.path.join(output_dir, f"{0}{name}.png")  # 拼接保存路径

    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 确保保存路径的目录存在

    # 将文件复制到保存路径
    shutil.copy2(video_path, save_path)  # 使用shutil模块的copy2函数复制文件
    print(f"Copied {video_path} to {save_path}")  # 打印文件复制成功的信息
