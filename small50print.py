from ultralytics import YOLO
import argparse
import os
import glob
import cv2  # OpenCV for image processing

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  # load an official model

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, required=True)
parser.add_argument('--images-dir', type=str, required=True)

args = parser.parse_args()

# Load Video List
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True)])

num_video = len(video_list)
print("Find ", num_video, " videos")

# Counter for deleted folders
deleted_folders_count = 0

# Process each video
for i in range(num_video):
    video_path = video_list[i]
    video_name_with_ext = os.path.basename(video_path)
    video_name = os.path.splitext(video_name_with_ext)[0]
    print(i, '/', num_video, video_name)

    images_dir = os.path.join(
        args.images_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)


    # Process each image in the directory
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))

    # Check if the number of images is less than 50 or if the directory is empty
    if len(image_files) < 50:
        print(f"The path {images_dir} contains less than 50 images or is empty and will be deleted.")

        # # Delete all images in the directory
        # for image_file in image_files:
        #     os.remove(image_file)

        # # Remove the directory if empty
        # if not os.listdir(images_dir):
        #     os.rmdir(images_dir)

        # # Delete the corresponding video file
        # os.remove(video_path)
        # print(f"Deleted corresponding video file: {video_path}")

        # Increase the counter for deleted folders
        deleted_folders_count += 1

# Print the total number of deleted folders
print(f"Total number of deleted folders: {deleted_folders_count}")
