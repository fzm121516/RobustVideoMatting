import torch
from model import MattingNetwork
import argparse
import os
import glob

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

from inference import convert_video

# convert_video(
#     model,  # 模型，可以加载到任何设备（cpu 或 cuda）
#     input_source='001-bg-01-090.avi',  # 视频文件，或图片序列文件夹
#     # input_resize=(1920, 1080),       # [可选项] 缩放视频大小
#     # downsample_ratio=0.25,           # [可选项] 下采样比，若 None，自动下采样至 512px
#     output_type='video',  # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
#     output_composition='com.mp4',  # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
#     output_alpha="pha.mp4",  # [可选项] 输出透明度预测
#     # output_foreground="fgr.mp4",  # [可选项] 输出前景预测
#     output_video_mbps=4,  # 若导出视频，提供视频码率
#     seq_chunk=12,  # 设置多帧并行计算
#     # num_workers=1,                   # 只适用于图片序列输入，读取线程
#     progress=True  # 显示进度条
# )
#


# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)

args = parser.parse_args()

# Load Images
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True)])

num_video = len(video_list)
print("Find ", num_video, " videos")

# Process
for i in range(num_video):
    video_path = video_list[i]
    video_name = video_path[video_path.rfind('/') + 1:video_path.rfind('.')]
    print(i, '/', num_video, video_name)

    # save results
    output_dir = args.result_dir + video_path[len(args.videos_dir):video_path.rfind('/')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 拼接保存路径并创建
    save_path = os.path.join(output_dir, video_name + '.avi')

    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    convert_video(
        model,  # 模型，可以加载到任何设备（cpu 或 cuda）
        input_source=video_path,  # 视频文件，或图片序列文件夹
        # input_resize=(1920, 1080),       # [可选项] 缩放视频大小
        # downsample_ratio=0.25,           # [可选项] 下采样比，若 None，自动下采样至 512px
        output_type='video',  # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
        # output_composition='com.mp4',  # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
        output_alpha=save_path,  # [可选项] 输出透明度预测
        # output_foreground="fgr.mp4",  # [可选项] 输出前景预测
        output_video_mbps=4,  # 若导出视频，提供视频码率
        seq_chunk=12,  # 设置多帧并行计算
        # num_workers=1,                   # 只适用于图片序列输入，读取线程
        progress=True  # 显示进度条
    )
