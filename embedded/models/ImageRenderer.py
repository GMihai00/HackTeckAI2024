import cv2
import threading
import queue
import uuid

class ImageRender:
    def __init__(self):
        self.image_queue = queue.Queue()
        self.is_running = False
        self.shutting_down = False
        self.cond_var_render = threading.Condition()
        self.thread_render = None
        self.id = uuid.uuid4()

    def load_image(self, image):
        with self.cond_var_render:
            self.image_queue.put(image)
            self.cond_var_render.notify()

    def start_rendering(self):
        if self.is_running:
            return False

        self.is_running = True
        self.shutting_down = False
        self.thread_render = threading.Thread(target=self.render_thread)
        self.thread_render.start()
        return True

    def render_thread(self):
        while not self.shutting_down:
            with self.cond_var_render:
                while self.image_queue.empty() and not self.shutting_down:
                    self.cond_var_render.wait()

                if self.shutting_down:
                    break

                # Get the next image from the queue
                image = self.image_queue.get()
                cv2.imshow("FRAME_" + str(self.id), image)
                
                # Wait for a short time to display the image
                # This is how I slow down overall video
                # for debug
                # cv2.waitKey(20)
                cv2.waitKey(1)

        cv2.destroyAllWindows()

    def stop_rendering(self):
        # print("Stop rendering")
        self.shutting_down = True
        try:
            self.cond_var_render.notify_all()
        except:
            with self.cond_var_render:
                self.cond_var_render.notify_all()
        
        # print("THREAD BUSTER")
        if self.thread_render and self.thread_render.is_alive():
            self.thread_render.join()

        self.is_running = False  # Update is_running here after thread has joined
        # print("DONE DONE ")

    def __del__(self):
        print("STOPPING RENDERING")
        self.stop_rendering()
        print("SOMETHING")
