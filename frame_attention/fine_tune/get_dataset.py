import cv2
import os
import time
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8n model for inference
model = YOLO('last.pt')  # Replace 'last.pt' with your actual YOLOv8n model path

# Update with your video path
video_path = 'path_to_video'
# Folder to save extracted frames
output_folder = 'frames_with_detections'
os.makedirs(output_folder, exist_ok=True)

# Open video capture
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Initialize variable to check if any container is detected
    container_detected = False
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())  # assuming only 1 class (container)

            # Filter detections by confidence
            if confidence >= 0.7:
                container_detected = True
                detections.append((class_id, confidence, x1, y1, x2, y2))
                # Optionally draw the detection on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # If container detected, save the frame and create an annotation file
    if container_detected:
        frame_filename = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_filename, frame)

        # Create annotation for YOLO format
        annotation_filename = f"{output_folder}/frame_{frame_count}.txt"
        with open(annotation_filename, 'w') as f:
            for det in detections:
                class_id, confidence, x1, y1, x2, y2 = det
                # YOLO format: class_id x_center y_center width height (normalized)
                x_center = (x1 + x2) / (2 * frame.shape[1])
                y_center = (y1 + y2) / (2 * frame.shape[0])
                width = (x2 - x1) / frame.shape[1]
                height = (y2 - y1) / frame.shape[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Move to the next frame
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Frames with detections saved in: {output_folder}")
