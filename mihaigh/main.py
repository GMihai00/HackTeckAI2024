from models.ObjectTracker import ObjectTracker
import time
import threading
import argparse
from models.MovingObjectGroup import MovingObjectGroup
import os

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--video_path", type=str, help="Path to video file to be processed", required=True)
    
    args = parser.parse_args()
    if not os.path.exists(args.video_path):
        print("Path not found!")
    
    # short edge cases "/home/mgherghinescu/projects/HackTeck2024/Dataset/Videori 240520/240520/240520_064129_064229.mp4"
    # 15 min long "/home/mgherghinescu/projects/HackTeck2024/Dataset/output_merged_video.mp4"
    # 3h video just for looking for issues "/home/mgherghinescu/projects/HackTeck2024/Dataset/mergedday1_3h.mp4"
    
    binTracker = ObjectTracker(args.video_path)
        
    binTracker.start_tracking(True);
    
    print("Program is running. Press Ctrl+C to stop.")
    start_time = time.time() 
    stop_event = threading.Event()  # Event to signal stop
        
    try:
        stop_event.wait()  # Wait indefinitely until the event is set
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
        
    binTracker.stop_tracking();
    
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\nProgram stopped by user. Time elapsed: {elapsed_time:.2f} seconds")
    
    print(f"TOTAL NUMBER OF BINS DETECTED: {MovingObjectGroup.BIN_COUNT - 1}")
    print("EXITING MAIN")
    

if __name__ == "__main__":
    main()