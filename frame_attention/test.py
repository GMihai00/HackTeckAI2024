import cv2
from ultralytics import YOLO

# Initialize the YOLOv8 model with the custom weights
model = YOLO('last.pt')  # Path to the YOLOv8 trained weights

# Load the video
video_path = 'videos/240520_062648_062748.mp4'
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare output video writer
output_video = 'output_detected_video.mp4'
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Loop through each detection and draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Bounding box coordinates as integers
            confidence = box.conf[0].cpu().numpy()  # Confidence score
            if confidence >= 0.5:  # Confidence threshold
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Display confidence score
                cv2.putText(frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                            2)

    # Write the frame with detections to the output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow('YOLOv8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
