import argparse
import os
import glob
from PIL import Image

# --------------- Arguments ---------------
parser = argparse.ArgumentParser(description='Test Images')
parser.add_argument('--videos-dir', type=str, required=True)
parser.add_argument('--images-dir', type=str, required=True)
parser.add_argument('--background-dir', type=str, required=True)
parser.add_argument('--result-dir', type=str, required=True)

args = parser.parse_args()

# Load Videos
video_list = sorted([*glob.glob(os.path.join(args.videos_dir, '**', '*.avi'), recursive=True)])

num_video = len(video_list)
print("Find ", num_video, " videos")

# Process
for i in range(num_video):
    video_path = video_list[i]
    video_name = os.path.basename(video_path)  # Get filename (including extension)
    print(i, '/', num_video, video_name)

    images_dir = os.path.join(
        args.images_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    background_path = os.path.join(
        args.background_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0],
        'background.png'  # Assuming background image is named 'background.png'
    )
    # Save results
    output_dir = os.path.join(
        args.result_dir,
        os.path.relpath(video_path, args.videos_dir).rsplit(os.sep, 1)[0]
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure the save path exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Load the background image
    background = Image.open(background_path).convert('RGBA')

    # Process each image
    for img_file in glob.glob(os.path.join(images_dir, '*.png')):
        img = Image.open(img_file).convert('RGBA')

        # Resize background to match the image size
        background_resized = background.resize(img.size, Image.Resampling.LANCZOS)

        # Composite image and background
        combined = Image.alpha_composite(background_resized, img)

        # Save the result
        save_path = os.path.join(output_dir, os.path.basename(img_file))
        combined.save(save_path, 'PNG')

        print(f'Processed {img_file}, saved to {save_path}')
