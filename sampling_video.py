import cv2
import os
from tqdm import tqdm

video_path = "./dataset/.custom/eval/eval01.mp4"
output_path = "./dataset/.custom/eval/samples"
samples = 60

if not os.path.exists(output_path):
    os.makedirs(output_path)

video = cv2.VideoCapture(video_path)

fps = 30
frame_count = 1800
frame_between_sample = int(frame_count/samples)

for i in tqdm(range(0, frame_count, frame_between_sample)):

    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = video.read()

    # Check if there was a problem reading the frame
    if not ret: break

    frame_filename = os.path.join(output_path, f"frame{i}.jpg")
    cv2.imwrite(frame_filename, frame)

video.release()

print(f"Done sampling {samples} images to {output_path}.")