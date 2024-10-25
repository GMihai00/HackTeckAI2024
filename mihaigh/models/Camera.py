import cv2
import threading
import time
from typing import Optional

class Camera:
    def __init__(self, id_: Optional[int] = None, path_video: Optional[str] = None):
        self.id_ = id_ if id_ is not None else -1
        self.path_video = path_video
        self.video_capture = None
        self.current_image = None
        self.shutting_down = False
        self.thread_read = None
        self.lock = threading.Lock()
        self.cond_var_read = threading.Condition(self.lock)

    def start(self) -> bool:
        if self.video_capture:
            print("Camera already powered on. Nothing to do")
            return False

        if self.path_video:
            self.video_capture = cv2.VideoCapture(self.path_video)
        else:
            self.video_capture = cv2.VideoCapture(self.id_)

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

    def _read_frames(self):
        while not self.shutting_down and self.video_capture and self.video_capture.isOpened():
            with self.lock:
                if self.current_image is not None:
                    self.cond_var_read.wait()

                ret, frame = self.video_capture.read()
                if ret:
                    frame = cv2.resize(frame, (854, 480))
                    self.current_image = frame
                    self.cond_var_read.notify()
                else:
                    print("Failed to read image.")
                    break

    def is_running(self) -> bool:
        with self.lock:
            return self.video_capture is not None

    def get_id(self) -> int:
        return self.id_

    def get_image(self) -> Optional[cv2.Mat]:
        with self.lock:
            if self.current_image is None:
                self.cond_var_read.notify()
                return None
            image_copy = self.current_image.copy()
            self.current_image = None
            self.cond_var_read.notify()
            return image_copy