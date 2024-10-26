import cv2
import threading
import queue
from concurrent.futures import Future, ThreadPoolExecutor

from ultralytics import YOLO

# dummy
class YOLOClient:

    def __init__(self):
        import os

        # Print the current working directory
        print("Current Directory:", os.getcwd())
        self.model = YOLO('./mihaigh/models/binmodel.pt')
        pass
        
    def connect(self, host, port):
        # Placeholder for connecting to TensorFlow server
        return True

    def get_bin_count_inside_image(self, image):

        results = self.model(image, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                confidence = float(box.conf[0].cpu().numpy())
                if confidence >= 0.69:
                    # print(f"CONFIDENCE SCORE: {confidence}")
                    detections.append([confidence])
        
        return len(detections)

class BinDetectClient:
    def __init__(self):
        self.tensorflow_client = YOLOClient()
        self.task_queue = queue.Queue()
        self.is_running = False
        self.shutting_down = False
        self.should_clear_all_tasks = False
        self.bin_count_task_map = {}
        self.cond_var_detect = threading.Condition()
        self.thread_detect = None
        self.executor = ThreadPoolExecutor()

        if not self.tensorflow_client.connect("TENSORFLOW_SERVER_HOST", "TENSORFLOW_SERVER_PORT"):
            raise RuntimeError("Failed to connect to TensorFlow server")

    def load_task(self, task_id, image):
        with self.cond_var_detect:
            self.task_queue.put((task_id, image))
            self.cond_var_detect.notify()
    
    def get_bins_present_in_image(self, image):
        # Returns a future representing the result of bin counting in the image
        return self.executor.submit(self.tensorflow_client.get_bin_count_inside_image, image)

    def start_detecting(self):
        if self.is_running:
            return False

        self.is_running = True
        self.shutting_down = False
        self.thread_detect = threading.Thread(target=self._detect_thread)
        self.thread_detect.start()
        return True

    def _detect_thread(self):
        while not self.shutting_down:
            with self.cond_var_detect:
                while self.task_queue.empty() and not self.shutting_down:
                    self.cond_var_detect.wait()

                if self.shutting_down:
                    break

                if self.should_clear_all_tasks:
                    while not self.task_queue.empty():
                        if self.shutting_down:
                            break

                        task_id, image = self.task_queue.get()
                        self.bin_count_task_map[task_id] = self.get_bins_present_in_image(image)
                else:
                    task_id, image = self.task_queue.get()
                    self.bin_count_task_map[task_id] = self.get_bins_present_in_image(image)

    def stop_detecting(self):
        self.shutting_down = True
        
        try:
            self.cond_var_detect.notify_all()
        except:
            with self.cond_var_detect:
                self.cond_var_detect.notify_all()
        if self.thread_detect and self.thread_detect.is_alive():
            self.thread_detect.join()

    def wait_for_finish(self):
        self.should_clear_all_tasks = True
        with self.cond_var_detect:
            self.cond_var_detect.notify_all()

        result = {}
        
        map_cpy = self.bin_count_task_map.copy()
        
        for task_id, future in map_cpy.items():
            result[task_id] = future.result()

        self.bin_count_task_map.clear()
        self.should_clear_all_tasks = False
        return result

    def __del__(self):
        print("STOPPING BIN CLIENT")
        self.stop_detecting()
        self.executor.shutdown()
        print("DONE STOPPING BIN CLIENT")
