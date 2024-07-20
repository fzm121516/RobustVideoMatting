import os
import glob
import cv2
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, required=True)  # 视频文件夹路径
parser.add_argument('--result-dir', type=str, required=True)   # 结果保存文件夹路径

# 解析命令行参数
args = parser.parse_args()

# 加载视频文件
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.mp4'), recursive=True)])

num_video = len(video_list)  # 视频数量
print("Find ", num_video, " videos")  # 输出找到的视频数量

# 处理每个视频
for i in range(num_video):
    video_path = video_list[i]  # 当前视频路径
    # video_name = os.path.basename(video_path)  # 获取文件名（包含扩展名）
    # 获取文件名（包含扩展名）
    video_name_with_ext = os.path.basename(video_path)

    # 去掉扩展名
    video_name = os.path.splitext(video_name_with_ext)[0]
    print(i, '/', num_video, video_name_with_ext)  # 输出处理进度

    # save results
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # 帧计数器

    while True:
        ret, frame = cap.read()  # 读取一帧
        if not ret:  # 如果没有读取到帧，结束循环
            break
        frame_count += 1  # 帧计数加1

        # 保存帧为图片
        save_path = os.path.join(output_dir, f"{video_name}-{frame_count:03d}.png")
        cv2.imwrite(save_path, frame)  # 保存当前帧为PNG格式

    cap.release()  # 释放视频对象

print("Processing complete!")  # 输出处理完成的信息
