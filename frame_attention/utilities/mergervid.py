import os
from moviepy.editor import VideoFileClip, concatenate_videoclips


def merge_videos_with_gpu(folder_path, output_filename="merged_video.mp4"):
    # List to store video clips
    video_clips = []

    # Loop through all files in the folder
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # Check if file is a video file (e.g., .mp4)
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"Adding {filename} to video clips")
            video_clips.append(VideoFileClip(file_path))

    # Concatenate all video clips
    if video_clips:
        final_clip = concatenate_videoclips(video_clips)

        # Use GPU-accelerated encoding
        final_clip.write_videofile(
            output_filename,
            codec="h264_nvenc",  # NVIDIA GPU codec for H.264 encoding
            ffmpeg_params=["-preset", "fast"],  # Adjust preset as needed
            audio_codec="aac"
        )

        print(f"Merged video saved as {output_filename}")
    else:
        print("No video files found in the specified folder.")


# Example usage
folder_path = "D:\\Alex\Desktop\\Heckathon\\Set date\\Videori 240520\\240520"
merge_videos_with_gpu(folder_path, "output_merged_video1.mp4")
