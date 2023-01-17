import os
from moviepy.editor import VideoFileClip

dir_in = "./dataset/dashcam/raw"
dir_out = "./dataset/dashcam/processed_720p"

print("Input directory: " + dir_in)
print("Output directory: " + dir_out)

if not os.path.isdir(dir_out):
  os.makedirs(dir_out)

for filename in os.listdir(dir_in):
    f = os.path.join(dir_in, filename)
    if os.path.isfile(f):
        print(f + " is being processed...")
        videoclip = VideoFileClip(f)
        new_clip = videoclip.without_audio() # Remove audio
        new_clip = new_clip.resize(height=720) # Resize video
        new_clip.write_videofile(os.path.join(dir_out, filename))

print("All done.")