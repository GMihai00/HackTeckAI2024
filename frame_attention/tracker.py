import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sort import Sort
import time
import os
import argparse
from Remover import remove

# Parse command-line arguments if provided
parser = argparse.ArgumentParser(description="Object tracking on a video.")
parser.add_argument('--video_path', type=str, default=r"videos/Video_jurizare1.mp4")
args = parser.parse_args()

# Set the video path from the argument or default value
video_path = args.video_path
filename = os.path.splitext(os.path.basename(video_path))[0]

# Create 'track_output' directory if it doesn't exist
output_dir = 'track_output'
os.makedirs(output_dir, exist_ok=True)

# Define paths for video and excel output in the track_output directory
result_video_path = os.path.join(output_dir, f"Track_of_{filename}.mp4")
excel_file_path = os.path.join(output_dir, f"{filename}_tracking_log.xlsx")
print(excel_file_path)

# Load the YOLO model and initialize the tracker
model = YOLO('last.pt')
tracker = Sort(max_age=50, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer with the updated path
out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
print(f"Saving result video as: {result_video_path}")

# Excel logging setup
log_data = []

# Start a timer at the beginning of the video
start_time = time.time()
previous_valid_container_count = 0
last_change_time = start_time

# Dictionary to store consecutive frame counts for each ID
consecutive_frame_counts = {}
# Set to store validated unique IDs
validated_containers = set()
# Counter for unique containers consecutive appearances
valid_container_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = remove.blur_people_in_frame(frame)
    results = model(frame)

    # Prepare detections array with correct shape
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())

            # Check if the detection comes with high confidence
            if confidence >= 0.9:
                # Calculate the perimeter of the bounding box
                perimeter = 2 * ((x2 - x1) + (y2 - y1))

                # Only add detection if perimeter is larger than 1500
                if perimeter > 1500:
                    detections.append([x1, y1, x2, y2, confidence])

    # Ensure detections array has the correct shape
    if detections:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))

    # Update SORT tracker with detections
    tracked_objects = tracker.update(detections)

    # Track IDs in current frame
    current_frame_ids = set()

    # Process each tracked object and count consecutive frames
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        current_frame_ids.add(track_id)

        # Draw bounding box and ID on the frame only if perimeter > 1500
        if (x2 - x1) * 2 + (y2 - y1) * 2 > 1500:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {valid_container_count}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update consecutive count for the current ID
            if track_id in consecutive_frame_counts:
                consecutive_frame_counts[track_id] += 1
            else:
                consecutive_frame_counts[track_id] = 1

            # Check if the ID qualifies as a valid container
            if (consecutive_frame_counts[track_id] >= 38 and
                    track_id not in validated_containers):
                validated_containers.add(track_id)
                valid_container_count += 1

    # Reset counts for IDs not detected in the current frame
    for track_id in list(consecutive_frame_counts.keys()):
        if track_id not in current_frame_ids:
            consecutive_frame_counts.pop(track_id)

    # Check for changes in valid_container_count
    if valid_container_count != previous_valid_container_count:
        # Log to Excel: "Bin", "Valid Container Count", "Start Time", "End Time"
        end_time = time.time()
        elapsed_start = last_change_time - start_time
        elapsed_end = end_time - start_time
        log_data.append(["Black Bin", valid_container_count, elapsed_start, elapsed_end])

        # Update the last change time and previous count
        last_change_time = end_time
        previous_valid_container_count = valid_container_count

    # Display container count on frame at the right side
    text = f'Containers Counted: {valid_container_count}'
    # Calculate text size to dynamically position it on the right
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = width - text_width - 10
    text_y = 30

    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display elapsed time since last container count change
    current_time = time.time()
    elapsed_since_change = current_time - last_change_time
    time_text = f'Time Since Last Detection: {elapsed_since_change:.1f} s'

    # Position this text below the container count
    cv2.putText(frame, time_text, (text_x, text_y + text_height + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow('CargoTrack analyzer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Write to Excel file
df = pd.DataFrame(log_data, columns=["Object type", "Track ID", "Start Time (s)", "End Time (s)"])
df.to_excel(excel_file_path, index=False)
print(f"Log saved to {excel_file_path}")
