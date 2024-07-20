from ultralytics import YOLO
import argparse
import os
import glob
import cv2  # OpenCV for image processing

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  # load an official model

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)

args = parser.parse_args()

# Load Images
images_list = sorted([*glob.glob(os.path.join(args.images_dir, '**', '*.png'), recursive=True)])

num_images = len(images_list)
print("Find ", num_images, " images")

# Process
for i in range(num_images):
    image_path = images_list[i]
    # 获取文件名（包含扩展名）
    images_name_with_ext = os.path.basename(image_path)

    # 去掉扩展名
    images_name = os.path.splitext(images_name_with_ext)[0]
    print(i, '/', num_images, images_name)

    # save results
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(image_path, args.images_dir).rsplit(os.sep, 1)[0]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 拼接保存路径并创建
    save_path = os.path.join(output_dir, images_name_with_ext)

    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Define path to the image file
    source = image_path

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    # Run inference on the source
    results = model(source)  # list of Results objects
    for result in results:
        for box in result.boxes:
            class_idx = int(box.cls.item())  # 将张量转换为整数索引
            class_name = model.names[class_idx]
            confidence = box.conf.item()  # 获取置信度

            if class_name == "person" and confidence > 0.7:
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
                cropped_save_path = os.path.join(output_dir, f"{images_name}_person_{i}.png")
                cv2.imwrite(cropped_save_path, cropped_image)
                print(f"Saved cropped image: {cropped_save_path}")

                print(f"Class: {class_name}, Confidence: {confidence}")
