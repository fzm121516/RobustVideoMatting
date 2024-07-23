from ultralytics import YOLO
import argparse
import os
import glob
import cv2  # OpenCV for image processing
from PIL import Image

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  # load an official model

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, required=True)
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)
args = parser.parse_args()

# Load Video List
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True)])

num_video = len(video_list)
print("Find ", num_video, " videos")

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
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the directory
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))

    # Step 1: Calculate confidence scores
    confidence_scores = []
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        results = model(image)  # list of Results objects
        total_confidence = 0
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls.item())  # Convert tensor to integer index
                class_name = model.names[class_idx]
                confidence = box.conf.item()  # Get confidence

                if class_name == "person":
                    total_confidence += confidence

        confidence_scores.append(total_confidence)

    # Step 2: Find the best window
    window_size = 50
    max_sum = 0
    max_start_index = 0

    # Compute initial window sum
    current_sum = sum(confidence_scores[:window_size])
    max_sum = current_sum

    for start in range(1, len(confidence_scores) - window_size + 1):
        current_sum = current_sum - confidence_scores[start - 1] + confidence_scores[start + window_size - 1]
        if current_sum > max_sum:
            max_sum = current_sum
            max_start_index = start

    # Get the best 50 images
    best_image_files = image_files[max_start_index:max_start_index + window_size]



    for idx, image_path in enumerate(best_image_files):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        output_image_path = os.path.join(output_dir, f"best_{idx:04d}.png")
        cv2.imwrite(output_image_path, image)
        print(f"Saved resized image to: {output_image_path}")
