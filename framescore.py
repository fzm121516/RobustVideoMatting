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
parser.add_argument('--videos-dir', type=str, required=True)
parser.add_argument('--images-dir', type=str, required=True)
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

    # Initialize counters and accumulators
    num_images_with_highest_person_confidence = 0
    total_images = len(image_files)
    total_confidence = 0

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

        # Update total confidence and check if "person" has the highest confidence
        total_confidence += highest_confidence
        if person_confidence == highest_confidence:
            num_images_with_highest_person_confidence += 1

    # Calculate and print the average confidence and fraction
    if total_images > 0:
        average_confidence = total_confidence / total_images
        fraction_person_highest_confidence = num_images_with_highest_person_confidence / total_images
        print(f"Average Confidence Score for {video_name}: {average_confidence:.4f}")
        print(
            f"Fraction of images where 'person' has the highest confidence in {video_name}: {fraction_person_highest_confidence:.4f}")
    else:
        print(f"No images processed for {video_name}")
