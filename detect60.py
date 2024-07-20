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

    # Get the best 60 images
    best_image_files = image_files[max_start_index:max_start_index + window_size]

    # Step 3: Crop and save the best images
    for idx, image_path in enumerate(best_image_files):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        results = model(image)  # list of Results objects
        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls.item())  # Convert tensor to integer index
                class_name = model.names[class_idx]
                confidence = box.conf.item()  # Get confidence

                if class_name == "person" :
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    crop_width, crop_height = 480, 960
                    crop_x1 = center_x - crop_width // 2
                    crop_y1 = center_y - crop_height // 2
                    crop_x2 = center_x + crop_width // 2
                    crop_y2 = center_y + crop_height // 2

                    # Adjust the crop box if it goes beyond the image boundaries
                    if crop_x1 < 0:
                        crop_x2 -= crop_x1  # Shift right
                        crop_x1 = 0
                    if crop_y1 < 0:
                        crop_y2 -= crop_y1  # Shift down
                        crop_y1 = 0
                    if crop_x2 > image.shape[1]:
                        crop_x1 -= (crop_x2 - image.shape[1])  # Shift left
                        crop_x2 = image.shape[1]
                    if crop_y2 > image.shape[0]:
                        crop_y1 -= (crop_y2 - image.shape[0])  # Shift up
                        crop_y2 = image.shape[0]

                    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Save the cropped image
                    base_name = os.path.basename(image_path)
                    cropped_save_path = os.path.join(output_dir, f"person_{idx}_{base_name}")
                    cv2.imwrite(cropped_save_path, cropped_image)
                    print(f"Saved cropped image: {cropped_save_path}")

                    print(f"Class: {class_name}, Confidence: {confidence}")
