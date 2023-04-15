import cv2
import os
from tqdm import tqdm

video_dir = "./dataset/.custom/eval/"
output_dir  = "./dataset/.custom/eval/samples"
samples = 60

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)) and f.endswith(".mp4") or f.endswith(".MP4")]

fps = 30
frame_count = 1800
frame_between_sample = int(frame_count/samples)

for video_number, video_file in enumerate(video_files):

    print(f"\nProcessing video {video_number+1} of {len(video_files)}...")
    video_path = os.path.join(video_dir, video_file)
    video = cv2.VideoCapture(video_path)

    frame_dir = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}")
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    file_number = 1
    for i in tqdm(range(0, frame_count, frame_between_sample)):

        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()

        # Check if there was a problem reading the frame
        if not ret: break
        
        frame_filename = os.path.join(frame_dir, f"{file_number}_frame{i}.jpg")
        cv2.imwrite(frame_filename, frame)
        file_number += 1

    video.release()

    print(f"Done sampling {samples} images from {video_file} to {frame_dir}")