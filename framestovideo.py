import os
import glob
import cv2
import argparse







# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, required=True)
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)
parser.add_argument('--fps', type=int, default=25, help='Frames per second for the output video')  # 视频帧率

args = parser.parse_args()

# Load Videos
video_list = sorted(glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True))

num_video = len(video_list)
print("Find ", num_video, " images")

# Process
for i in range(num_video):
    video_path = video_list[i]
    # 获取文件名（包含扩展名）
    video_name_with_ext = os.path.basename(video_path)

    # 去掉扩展名
    video_name = os.path.splitext(video_name_with_ext)[0]
    print(i, '/', num_video, video_name)

    images_dir = os.path.join(
        args.images_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    # Save results
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 拼接保存路径并创建
    # save_path = os.path.join(output_dir, video_name)
    save_path = output_dir


    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image_list = sorted(glob.glob(os.path.join(images_dir, '*.png')))

    num_images = len(image_list)  # 图片数量
    print("Find ", num_images, " images in " ,images_dir)  # 输出找到的图片数量

    if num_images == 0:
        print("No images found in the specified directory.")
        exit()

    # 获取第一张图片的尺寸
    first_image = cv2.imread(image_list[0])
    height, width, layers = first_image.shape

    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_video_path = os.path.join(output_dir, f"{video_name}.mp4")
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    # 创建视频写入对象
    out = cv2.VideoWriter(output_video_path, fourcc, args.fps, (width, height))

    # 处理每张图片
    for i, image_path in enumerate(image_list):
        print(i, '/', num_images, image_path)  # 输出处理进度
        frame = cv2.imread(image_path)  # 读取图片
        out.write(frame)  # 写入视频帧

    out.release()  # 释放视频写入对象

    print("Processing complete! Video saved to:", output_video_path)  # 输出处理完成的信息
