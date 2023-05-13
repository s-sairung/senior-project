from moviepy.editor import *

dir = "./dataset/.custom/drive1/selected"

file1 = "NO20230412-144307-000140.MP4"
file2 = "NO20230412-144407-000141.MP4"

clip1_start = -14
clip2_end = 46

if -1*clip1_start + clip2_end != 60:
    raise Exception

clip1 = VideoFileClip(os.path.join(dir, file1))
clip2 = VideoFileClip(os.path.join(dir, file2))

clip1 = clip1.subclip(clip1_start, 60)
clip2 = clip2.subclip(0, clip2_end)
 
final = concatenate_videoclips([clip1, clip2])
final.write_videofile(os.path.join(dir, file1[:-4]) + "_merged.MP4")
