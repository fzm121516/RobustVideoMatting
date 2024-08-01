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
parser.add_argument('--videos-dir', type=str,  default="/home/fanzheming/zm/mygait/datasets/CASIA-B/dataset-b3")
parser.add_argument('--images-dir', type=str, default="/home/fanzheming/zm/mygait/datasets/CASIA-B/dataset-b-frames-50-original-1280960")
args = parser.parse_args()

# Load Video List
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True)])

num_video = len(video_list)
print("Find ", num_video, " videos")

# Initialize global counters and accumulators
total_images = 0
total_confidence = 0
num_images_with_highest_person_confidence = 0

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

    # Process each image in the directory
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))

    parts = video_name.split('-')

    if len(parts) == 4:  # If the number of parts is 4, the filename format is correct
        gait_id = parts[0]
        gait_type = f"{parts[1]}-{parts[2]}"
        gait_view = parts[3]  # Combine the second and third parts
    else:  # If the filename format is not as expected, skip this file
        print(f"Unexpected filename format: {video_name}")
        continue

    # Check if gait_type is in allowed_gait_types
    if gait_type not in allowed_gait_types:
        print(f"Gait type {gait_type} not in allowed list, skipping.")
        continue

    # Check if gait_id is within the range 075 to 124
    try:
        gait_id_num = int(gait_id)
        if gait_id_num < 75 or gait_id_num > 124:
            print(f"Gait ID {gait_id} not in the allowed range (075-124), skipping.")
            continue
    except ValueError:
        print(f"Invalid Gait ID {gait_id}, skipping.")
        continue

    # Process images for the current video
    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        results = model(image)  # list of Results objects
        highest_confidence = 0
        person_confidence = 0

        for result in results:
            for box in result.boxes:
                class_idx = int(box.cls.item())  # Convert tensor to integer index
                class_name = model.names[class_idx]
                confidence = box.conf.item()  # Get confidence

                if class_name == "person":
                    person_confidence = confidence  # Track confidence for "person"

                if confidence > highest_confidence:
                    highest_confidence = confidence

        # Update global counters and accumulators
        total_confidence += highest_confidence
        if person_confidence == highest_confidence:
            num_images_with_highest_person_confidence += 1

        total_images += 1

# Calculate and print the average confidence and fraction across all images
if total_images > 0:
    average_confidence = total_confidence / total_images
    fraction_person_highest_confidence = num_images_with_highest_person_confidence / total_images
    print(f"Average Confidence Score across all images: {average_confidence:.4f}")
    print(f"Fraction of images where 'person' has the highest confidence across all images: {fraction_person_highest_confidence:.4f}")
else:
    print("No images processed.")
