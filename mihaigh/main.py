from models.ObjectTracker import ObjectTracker
import time

def main():

    binTracker = ObjectTracker("/home/mgherghinescu/projects/HackTeck2024/Dataset/Videori 240520/240520/240520_055427_055527.mp4")
    
    binTracker.start_tracking(True);
    
    print("Program is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)  # Keeps the program alive and responsive
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
        
    binTracker.stop_tracking();
    
    print("EXITING MAIN")
    

if __name__ == "__main__":
    main()