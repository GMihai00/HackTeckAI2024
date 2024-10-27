import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load the YOLO model and tracker
model = YOLO('last.pt')
tracker = Sort(max_age=50, min_hits=2, iou_threshold=0.3)

video_path = 'videos/mergedday1_3h.mp4'

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = video_path + '_tracked.mp4'
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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

    results = model(frame)

    # Prepare detections array with correct shape
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            if confidence >= 0.5:
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

        # Draw bounding box and ID on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Check or update consecutive count for the current ID
        if track_id in consecutive_frame_counts:
            consecutive_frame_counts[track_id] += 1
        else:
            consecutive_frame_counts[track_id] = 1

        # Check if the ID qualifies as a valid container
        if (consecutive_frame_counts[track_id] >= 35 and
                track_id not in validated_containers):
            validated_containers.add(track_id)
            valid_container_count += 1

    # Reset counts for IDs not detected in the current frame
    for track_id in list(consecutive_frame_counts.keys()):
        if track_id not in current_frame_ids:
            consecutive_frame_counts.pop(track_id)

    # Display container count on frame
    cv2.putText(frame, f'Containers Counted: {valid_container_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)
    cv2.imshow('YOLOv8n Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
