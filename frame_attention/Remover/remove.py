import cv2
from ultralytics import YOLO

model = YOLO("yolo11x.pt")


def blur_people_in_frame(frame, model=model, confidence_threshold=0.2, blur_ksize=(51, 51)):
    """
    Detects people in the given frame and applies Gaussian blur to their bounding boxes.

    Args:
        frame (ndarray): The input video frame.
        model (YOLO): The YOLO model object.
        confidence_threshold (float): Minimum confidence threshold for person detection.
        blur_ksize (tuple): Kernel size for the Gaussian blur.

    Returns:
        ndarray: The frame with blurred people regions.
    """
    # Run the YOLO model to detect objects in the frame
    results = model(frame)

    # Loop through each detection and apply blurring to bounding boxes for the 'person' class only
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Bounding box coordinates as integers
            confidence = box.conf[0].cpu().numpy()  # Confidence score
            class_id = int(box.cls[0].cpu().numpy())  # Class ID of the detection

            # Check if the detection is of class 'person' (class ID 0 in COCO dataset)
            if class_id == 0 and confidence >= confidence_threshold:
                # Extract the region of interest (ROI) for blurring
                person_region = frame[y1:y2, x1:x2]
                # Apply a Gaussian blur to the detected person region
                blurred_region = cv2.GaussianBlur(person_region, blur_ksize, 0)
                # Place the blurred region back into the frame
                frame[y1:y2, x1:x2] = blurred_region

    return frame

# Just for testing purposes
# # Load the video
# video_path = '240520_055823_055923.mp4'
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # Prepare output video writer
# output_video = 'output_detected_video_blurred.mp4'
# out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#
# # Process the video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Blur people in the current frame
#     frame = blur_people_in_frame(frame, model)
#
#     # Write the frame with blurred detections to the output video
#     out.write(frame)
#
#     # Optional: Display the frame (press 'q' to quit)
#     cv2.imshow('YOLOv11x Detection with Blurred Bounding Box', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()
