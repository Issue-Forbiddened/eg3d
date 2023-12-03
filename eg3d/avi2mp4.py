from moviepy.editor import VideoFileClip
import os

# File paths
avi_file_path = '/home1/jo_891/data1/eg3d/eg3d/control3diff_trained_clip_mv/output_video_290001_additional_sample_prompt_a_young_male.avi'
mp4_file_path = avi_file_path.replace('.avi', '.mp4')

# Convert AVI to MP4
clip = VideoFileClip(avi_file_path)
clip.write_videofile(mp4_file_path, codec='libx264')

# gif_file_path = avi_file_path.replace('.avi', '.gif')

# # Convert AVI to GIF
# clip = VideoFileClip(avi_file_path)
# clip.write_gif(gif_file_path,fps=25)


