import cv2
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (yolov8n is the nano version for speed, you can also use yolov8s or yolov8m)
model = YOLO('last.pt')  # you can replace yolov8n.pt with yolov8s.pt or yolov8m.pt for more accuracy

# Function to run detection on an image
def detect_on_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)

    # Perform detection
    results = model(img)

    # The `results` variable contains a list of result objects, we'll take the first result and plot it.
    detected_img = results[0].plot()  # This returns an image with bounding boxes drawn

    # Display the results using OpenCV
    cv2.imshow('YOLOv8n Image Detection', detected_img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the image windo

# Main function to switch between image or video detection
if __name__ == "__main__":
    # You can replace these paths with any image or video path
    image_path = "small2.png"  # Replace with your image path

    detect_on_image(image_path)
    # detect_on_video(video_path)
