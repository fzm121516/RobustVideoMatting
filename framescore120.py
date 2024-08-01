from ultralytics import YOLO
import argparse
import os
import glob
import cv2  # OpenCV for image processing

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")  # load an official model
allowed_gait_types = ['nm-05', 'nm-06', 'bg-01', 'bg-02', 'cl-01', 'cl-02']

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, default="/home/fanzheming/zm/mygait/datasets/CASIA-B/dataset-b3")
parser.add_argument('--images-dir', type=str, default="/home/fanzheming/zm/mygait/datasets/CASIA-B/dataset-b-frames-50-original-480960")
args = parser.parse_args()

# Load Video List
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True)])

num_video = len(video_list)
print("Find ", num_video, " videos")

# Initialize global counters and accumulators
total_images = 0
total_confidence = 0
num_images_with_highest_person_confidence = 0
num_images_with_person_detected = 0  # Counter for images with person detected

# Process each video
for i in range(num_video):
    video_path = video_list[i]
    video_name_with_ext = os.path.basename(video_path)
    video_name = os.path.splitext(video_name_with_ext)[0]

    images_dir = os.path.join(
        args.images_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )

    # Process each image in the directory
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))

    parts = video_name.split('-')

    if len(parts) == 4:  # If the number of parts is 4, the filename format is correct
        gait_id = parts[0]
        gait_type = f"{parts[1]}-{parts[2]}"
        gait_view = parts[3]  # Combine the second and third parts
    else:  # If the filename format is not as expected, skip this file
        continue

    # Check if gait_type is in allowed_gait_types
    if gait_type not in allowed_gait_types:
        continue

    # Check if gait_id is within the range 075 to 124
    try:
        gait_id_num = int(gait_id)
        if gait_id_num < 75 or gait_id_num > 124:
            continue
    except ValueError:
        continue

    # Process images for the current video
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Resize the image to 120x240
        resized_image = cv2.resize(image, (120, 240))

        results = model(resized_image)  # list of Results objects
        highest_confidence = 0
        person_confidence = 0
        person_detected = False

        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls.item())  # Convert tensor to integer index
                class_name = model.names[class_idx]
                confidence = box.conf.item()  # Get confidence

                if class_name == "person":
                    person_confidence = confidence  # Track confidence for "person"
                    person_detected = True

                if confidence > highest_confidence:
                    highest_confidence = confidence

        if not person_detected:
            print(f"No person detected in this image: {image_path}")

        # Update global counters and accumulators
        total_confidence += highest_confidence
        if person_confidence == highest_confidence:
            num_images_with_highest_person_confidence += 1

        if person_detected:
            num_images_with_person_detected += 1

        total_images += 1

# Calculate and print the average confidence and fraction across all images
if total_images > 0:
    average_confidence = total_confidence / total_images
    fraction_person_detected = num_images_with_person_detected / total_images
    print(f"Average Confidence Score across all images: {average_confidence:.4f}")
    print(f"Fraction of images where 'person' is detected: {fraction_person_detected:.4f}")
else:
    print("No images processed.")
