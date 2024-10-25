import cv2
import threading
import queue
from concurrent.futures import Future, ThreadPoolExecutor
# import tensorflow as tf  # placeholder for tensorflowClient implementation

# dummy
class TensorFlowClient:
    def connect(self, host, port):
        # Placeholder for connecting to TensorFlow server
        return True

    def get_car_count_inside_image(self, image):
        # Placeholder for TensorFlow model to count cars in the image always return 1 car
        return 1

class BinDetectClient:
    def __init__(self):
        self.tensorflow_client = TensorFlowClient()
        self.task_queue = queue.Queue()
        self.is_running = False
        self.shutting_down = False
        self.should_clear_all_tasks = False
        self.car_count_task_map = {}
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
        # Returns a future representing the result of car counting in the image
        return self.executor.submit(self.tensorflow_client.get_car_count_inside_image, image)

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
                        self.car_count_task_map[task_id] = self.get_bins_present_in_image(image)
                else:
                    task_id, image = self.task_queue.get()
                    self.car_count_task_map[task_id] = self.get_bins_present_in_image(image)

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
        for task_id, future in self.car_count_task_map.items():
            result[task_id] = future.result()

        self.car_count_task_map.clear()
        self.should_clear_all_tasks = False
        return result

    def __del__(self):
        print("STOPPING BIN CLIENT")
        self.stop_detecting()
        self.executor.shutdown()
        print("DONE STOPPING BIN CLIENT")
