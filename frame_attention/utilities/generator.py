import cv2
import os


def extract_frames(video_path, output_folder, num_frames, frame_size=(640, 640)):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)  # Calculate the interval between frames

    extracted_frames = 0
    current_frame = 0

    while extracted_frames < num_frames and video_capture.isOpened():
        # Set the frame position
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Read the frame
        ret, frame = video_capture.read()

        if not ret:
            break

        # Resize the frame
        frame_resized = cv2.resize(frame, frame_size)

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{extracted_frames + 1}.jpg")
        cv2.imwrite(frame_filename, frame_resized)

        extracted_frames += 1
        current_frame += frame_interval

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {extracted_frames} frames to '{output_folder}'.")


# Parameters
video_path = '240520_060514_060614.mp4'  # Path to the input video file
output_folder = 'extracted_frames'  # Output folder for frames
num_frames = 1000  # Number of frames to extract

extract_frames(video_path, output_folder, num_frames)