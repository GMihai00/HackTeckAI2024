from models.ObjectTracker import ObjectTracker
import time
import threading
import argparse
from models.MovingObjectGroup import MovingObjectGroup
import os
        
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_path", type=str, help="Path to video file to be processed", required=True)
    parser.add_argument("--enable_ocr", type=bool, help="Enable additional ocr processing of timestamp within video", default=False)
    parser.add_argument("--draw_moving", type=bool, help="Enable flag to draw all moving objects bounding boxes", default=False)
    parser.add_argument("--render_video", type=bool, help="If enabled display video content, to be disabled when running in prod", default=True)
    parser.add_argument("--render_post_processed_video", type=bool, help="Learning flag, to understand the applied video processing operations.", default=False)
    
    args = parser.parse_args()
    if not os.path.exists(args.video_path):
        print("Path not found!")
    
    
    # short edge cases "/home/mgherghinescu/projects/HackTeck2024/Dataset/Videori 240520/240520/240520_064129_064229.mp4"
    # 15 min long "/home/mgherghinescu/projects/HackTeck2024/Dataset/output_merged_video.mp4"
    # 3h video just for looking for issues "/home/mgherghinescu/projects/HackTeck2024/Dataset/mergedday1_3h.mp4"
    
    print("Program is running. Press Ctrl+C to stop.")
    start_time = time.time() 
    stop_event = threading.Event()  # Event to signal stop
    
    binTracker = ObjectTracker(args.video_path, stop_event, args.enable_ocr, args.draw_moving, args.render_post_processed_video)
        
    binTracker.start_tracking(args.render_video);
        
    try:
        stop_event.wait()  # Wait indefinitely until the event is set
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
        
    binTracker.stop_tracking();
    
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\nProgram stopped by user. Time elapsed: {elapsed_time:.2f} seconds")
    
    print(f"Total number of bins detected: {MovingObjectGroup.BIN_COUNT - 1}")

if __name__ == "__main__":
    main()