from models.ObjectTracker import ObjectTracker
import time
import threading

def main():
    # short edge cases
    binTracker = ObjectTracker("/home/mgherghinescu/projects/HackTeck2024/Dataset/Videori 240520/240520/240520_064129_064229.mp4")
    # 15 min long in 46 sec
    # binTracker = ObjectTracker("/home/mgherghinescu/projects/HackTeck2024/Dataset/output_merged_video.mp4")
    
    # 3h video just for looking for issues
    # binTracker = ObjectTracker("/home/mgherghinescu/projects/HackTeck2024/Dataset/mergedday1_3h.mp4")
    
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
    print("EXITING MAIN")
    

if __name__ == "__main__":
    main()