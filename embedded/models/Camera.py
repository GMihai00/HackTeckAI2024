import cv2
import threading
import time
from typing import Optional
import numpy as np

from .TimestampExtractorOCR import TimestampExtractorOCR

class Camera:
    def __init__(self, id_: Optional[int] = None, path_video: Optional[str] = None, enable_ocr=False):
        self.id_ = id_ if id_ is not None else -1
        self.path_video = path_video
        self.video_capture = None
        self.current_image = None
        self.shutting_down = False
        self.thread_read = None
        self.mutex_camera = threading.Lock()
        self.current_timestamp = 0
        self.cond_var_camera = threading.Condition(self.mutex_camera)
        self.lock = threading.Lock()
        self.cond_var_read = threading.Condition(self.lock)
        self.calc_timestamps = [0.0]
        self.time_stamp_extractor = None
        if enable_ocr:
            self.time_stamp_extractor = TimestampExtractorOCR()

    def start(self) -> bool:
        if self.video_capture:
            print("Camera already powered on. Nothing to do")
            return False

        if self.path_video:
            self.video_capture = cv2.VideoCapture(self.path_video)
        else:
            self.video_capture = cv2.VideoCapture(self.id_)
        
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                
        self.shutting_down = False
        self.thread_read = threading.Thread(target=self._read_frames)
        self.thread_read.start()
        print("Camera started")
        return True

    def stop(self):
        with self.lock:
            self.shutting_down = True
            self.cond_var_read.notify()
        
        if self.thread_read:
            self.thread_read.join()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        print("Camera stopped")
    
    def update_current_timestamp(self, frame):
    
        if self.time_stamp_extractor == None:
            timestamp  = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
            self.calc_timestamps.append(self.calc_timestamps[-1] + 1000/self.fps)
            
            self.current_timestamp += np.uint8(abs(timestamp - self.calc_timestamps[-1]))
        else:
            self.current_timestamp = self.time_stamp_extractor.extract_timestamp(frame)
        
    def _read_frames(self):
        while not self.shutting_down and self.video_capture and self.video_capture.isOpened():
            with self.cond_var_camera:
                # print("LOCK1")
                if self.current_image is not None and self.video_capture and self.video_capture.isOpened():
                    # print("IMAGE ALREADY EXISTING WAITING FOR READ!")
                    try:
                        self.cond_var_read.notify()
                    except:
                        with self.cond_var_read:
                            self.cond_var_read.notify()
                    self.cond_var_camera.wait()

                ret, frame = self.video_capture.read()
                if ret:
                    frame = cv2.resize(frame, (640, 640))
                    self.current_image = frame
                    
                    self.update_current_timestamp(frame)
                    
                    with self.cond_var_read:
                        self.cond_var_read.notify()
                else:
                    print("Failed to read image.")
                    try:
                        self.cond_var_read.notify()
                    except:
                        with self.cond_var_read:
                            self.cond_var_read.notify()
                    break
        self.shutting_down = True

    def is_running(self) -> bool:
        with self.lock:
            return self.video_capture is not None

    def get_id(self) -> int:
        return self.id_

    def get_image_and_timestamp(self) -> Optional[cv2.Mat]:
        with self.cond_var_read:
            # print("LOCK2")
            if self.current_image is None:
                # print("Current image is none")
                try:
                    self.cond_var_camera.notify()
                except:
                    with self.cond_var_camera:
                        self.cond_var_camera.notify()
                if self.shutting_down == False:
                    self.cond_var_read.wait()
            
            if self.current_image is None:
                print("Reached the end of the video")
                return None, None
                
            # print("Getting image")
            image_copy = self.current_image.copy()
            timestamp = self.current_timestamp
            self.current_image = None
            try:
                self.cond_var_camera.notify()
            except:
                with self.cond_var_camera:
                    self.cond_var_camera.notify()
            return image_copy, timestamp
            
    def __del__(self):
        print("Deleting camera")