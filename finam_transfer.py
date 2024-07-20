from ultralytics import YOLO
import argparse
import os
import glob
import cv2  # OpenCV for image processing
import numpy as np

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  # load an official model

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, required=True)
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--newimages-dir', type=str, required=True)
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
    newimages_dir = os.path.join(
        args.newimages_dir,
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
    new_image_files = sorted(glob.glob(os.path.join(newimages_dir, '*.jpg')) + glob.glob(os.path.join(newimages_dir, '*.png')))

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

    # Step 3: Resize and replace the best images
    target_width, target_height = 480, 960

    for idx, (image_path, new_image_path) in enumerate(zip(best_image_files, new_image_files)):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        new_image = cv2.imread(new_image_path, cv2.IMREAD_UNCHANGED)
        if new_image is None:
            print(f"Failed to load new image: {new_image_path}")
            continue

        # Convert images to RGBA if they are not already
        if image.shape[2] == 3:
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            image_rgba = image

        if new_image.shape[2] == 3:
            new_image_rgba = cv2.cvtColor(new_image, cv2.COLOR_BGR2BGRA)
        else:
            new_image_rgba = new_image

        # Convert image back to BGR for model inference
        image_bgr = cv2.cvtColor(image_rgba, cv2.COLOR_BGRA2BGR)
        results = model(image_bgr)  # list of Results objects
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls.item())  # Convert tensor to integer index
                class_name = model.names[class_idx]
                confidence = box.conf.item()  # Get confidence

                if class_name == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox_height = y2 - y1
                    bbox_width = bbox_height / 2  # Set width to half of height

                    # Adjust bounding box to center crop with new dimensions
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    new_x1 = int(center_x - bbox_width / 2)
                    new_y1 = int(center_y - bbox_height / 2)
                    new_x2 = int(center_x + bbox_width / 2)
                    new_y2 = int(center_y + bbox_height / 2)

                    # Ensure new coordinates are within image bounds
                    new_x1 = max(new_x1, 0)
                    new_y1 = max(new_y1, 0)
                    new_x2 = min(new_x2, image.shape[1])
                    new_y2 = min(new_y2, image.shape[0])

                    # # Resize the new image to fit the bounding box region
                    # resized_new_image = cv2.resize(new_image_rgba, (new_x2 - new_x1, new_y2 - new_y1))
                    #
                    # # Alpha blend the resized new image into the original image
                    # alpha_s = resized_new_image[:, :, 3] / 255.0
                    # alpha_l = 1.0 - alpha_s
                    #
                    # for c in range(0, 3):
                    #     image_rgba[new_y1:new_y2, new_x1:new_x2, c] = (alpha_s * resized_new_image[:, :, c] +
                    #                                                   alpha_l * image_rgba[new_y1:new_y2, new_x1:new_x2, c])
                    # Resize the new image to fit the bounding box region
                    resized_new_image = cv2.resize(new_image_rgba, (new_x2 - new_x1, new_y2 - new_y1))

                    # 将调整后的新图像完全替换原始图像中指定区域的内容
                    image_rgba[new_y1:new_y2, new_x1:new_x2, :] = resized_new_image

                    # Save the final image
                    base_name = os.path.basename(image_path)
                    final_save_path = os.path.join(output_dir, f"person-{idx:04d}.png")
                    cv2.imwrite(final_save_path, image_rgba)
                    print(f"Saved final image: {final_save_path}")

                    print(f"Class: {class_name}, Confidence: {confidence}")


