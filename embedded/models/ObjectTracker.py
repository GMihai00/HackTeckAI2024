import cv2
import numpy as np
import threading
from queue import Queue
from typing import Optional, List, Tuple

from .ImagePreprocesor import ImageProcessor

from .ImageRenderer import ImageRender
from .BinDetectClient import BinDetectClient

from .Camera import Camera
from .MovingObjectGroup import MovingObjectGroup

from .Utils import *

from .MovingObject import *

class ObjectTracker:
    def __init__(self, camera_source, stop_event=None):
        self.camera = Camera(path_video = camera_source)  # Can be camera ID or video path
        self.image_processor = ImageProcessor()
        self.image_render = ImageRender()
        self.bin_detector = BinDetectClient()
        self.image_queue = Queue()
        self.first_image_frame = None
        self.second_image_frame = None
        self.first_image_timestamp = None
        self.second_image_timestamp = None
        self.should_render = False
        self.task_id_to_obj_group = {}
        self.task_id = 0
        self.moving_objects = []
        self.horizontal_line_position = None
        self.crossing_line_left = [(0, 0), (0, 0)]
        self.crossing_line_right = [(0, 0), (0, 0)]
        self.mutex_camera = threading.Lock()
        self.mutex_process = threading.Lock()
        self.cond_var_camera = threading.Condition(self.mutex_camera)
        self.cond_var_process = threading.Condition(self.mutex_process)
        self.thread_camera = None
        self.thread_process = None
        self.stop_event = stop_event

    def wait_for_first_frame_appearance(self):
        with self.cond_var_process:
            while self.image_queue.empty() and self.camera.is_running():
                self.cond_var_process.wait()

            if not self.camera.is_running():
                return

            self.first_image_frame, self.first_image_timestamp = self.image_queue.get()
            self.setup_lines()
    
    def object_blocking_camera(self, color_percentage):
        if color_percentage > 90:
            # print("SKIPPING frame due to high similarity.")
            return True
        
        return False
    
    def process_images(self):
        self.wait_for_first_frame_appearance()
        while self.camera.is_running() or not self.image_queue.empty():
            with self.cond_var_process:
                if self.image_queue.empty() and self.camera.is_running():
                    self.cond_var_process.wait()

                if not self.camera.is_running():
                    # print("SHUTTING DONW OBJECT TRACKER!")
                    break
                
                # print("Fetching image")
                self.second_image_frame, self.second_image_timestamp = self.image_queue.get()

                # skip frames that have 90%  pixels the same (SOMETHING BLOCKING THE CAMERA)
                
                average_color = get_average_color(self.second_image_frame)
                similarity_percentage = calculate_color_similarity(self.second_image_frame, average_color)
                
                # print(f"COLOR SIMILARITY: {similarity_percentage}")
                if self.object_blocking_camera(similarity_percentage):
                    if self.should_render:
                        self.image_render.load_image(self.second_image_frame)
                        self.image_render.start_rendering()
                    continue
                
                img_threshold = self.image_processor.get_processed_merged_image(
                    self.first_image_frame, self.second_image_frame)
                
                # print("Detecting")   
                detected_moving_objs = self.image_processor.get_moving_objects_from_img(img_threshold)
                
                # print("Processing detected objects")
                if not self.moving_objects:
                    for moving_obj in detected_moving_objs:
                        self.add_new_moving_object(moving_obj)
                else:
                    self.match_found_obj_to_existing_ones(detected_moving_objs)
                
                # print("Done processing")
                
                cpy = self.second_image_frame.copy()
                self.draw_results_on_image(cpy)
                
                # print("DONE DRAWING")
                
                
                if self.should_render:
                    # movement only
                    # self.image_render.load_image(img_threshold)
                    # self.image_render.start_rendering()
                    
                    # all image
                    self.image_render.load_image(cpy)
                    self.image_render.start_rendering()
                
                detected_moving_objs.clear()
                self.first_image_frame = self.second_image_frame.copy()
                
                # print("DOne with everything")

    def start_tracking(self, should_render=False):
        if self.camera.start():
            self.should_render = should_render
            self.thread_camera = threading.Thread(target=self.camera_capture)
            self.thread_process = threading.Thread(target=self.process_images)
            self.thread_camera.start()
            self.thread_process.start()
            return True
        else:
            print("Failed to start tracking")
            return False

    def camera_capture(self):
        while self.camera.is_running():      
            # print("Getting image")
            image, timestamp = self.camera.get_image_and_timestamp()
            if image is not None:
                with self.cond_var_process:  # Acquires the lock for cond_var_process
                    # print("Image captured and added to queue")

                    self.image_queue.put((image, timestamp))
                    # print(f"SIZE QUEUE !!! {self.image_queue.qsize()}")
                    self.cond_var_process.notify()  # Notifies waiting threads
            else:
                # print("NO IMAGE TO FETCH!!!!!!!!!!!!!!!")
                break
        self.camera.stop()
        try:
            self.cond_var_process.notify()
        except:
            with self.cond_var_process:
                self.cond_var_process.notify()
        self.stop_event.set()


    def stop_tracking(self):
        # print("Stopping tracking!!!")
        self.camera.stop()
        try:
            self.cond_var_camera.notify()
        except:
            with self.cond_var_camera:  # Acquires the lock for cond_var_camera
                self.cond_var_camera.notify()  # Notifies waiting threads
        
        self.image_render.stop_rendering()
        self.image_queue.queue.clear()
        self.first_image_frame = None
        self.second_image_frame = None
        
        # print("STOPPPPP")
        try:
            self.cond_var_process.notify()
        except: 
            with self.cond_var_process:  # Acquires the lock for cond_var_process
                self.cond_var_process.notify()  # Notifies waiting threads
        # print("STOPPING THREADSS")
        if self.thread_process and self.thread_process.is_alive():
            self.thread_process.join()
            
        # print("STOPPING NEXT THREADS")
        if self.thread_camera and self.thread_camera.is_alive():
            self.thread_camera.join()
            
        # print("Stopping bin client")
        
        self.bin_detector.stop_detecting()
        
        print("DONE")


    def setup_lines(self):
        height, width = self.first_image_frame.shape[:2]
        self.horizontal_line_position = height // 2
        self.crossing_line_right = [(width // 2, int(self.horizontal_line_position * 0.75)),
                                    (width - 1, int(self.horizontal_line_position * 0.75))]
        self.crossing_line_left = [(0, self.horizontal_line_position), (width // 2 - 1, self.horizontal_line_position)]

    def draw_results_on_image(self, img):
        self.draw_obj_info_on_image(img, self.moving_objects)
        
        bin_count_map = self.bin_detector.wait_for_finish()
        for task_id, nr_bins in bin_count_map.items():
            if task_id in self.task_id_to_obj_group:
                self.task_id_to_obj_group[task_id].update_bin_state(nr_bins, self.second_image_timestamp)

        self.task_id_to_obj_group.clear()
        self.task_id = 0

    def draw_obj_info_on_image(self, img, moving_objects):
        cnt = 0
        for moving_obj_group in moving_objects:
            moving_obj = moving_obj_group.get_last_state()
            
            if moving_obj:
                if moving_obj_group.bin_id != 0:
                    color = (0, 0, 255)  # SCALAR_RED in BGR
                    # Draw bounding rectangle
                    cv2.rectangle(img, moving_obj.get_bounding_rect(), color, 2)
                    
                    # Set font parameters
                    font_face = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = (img.shape[0] * img.shape[1]) / 300000.0
                    font_thickness = int(round(font_scale * 1.0))
                    
                    # Get center position for text placement
                    center_position = moving_obj_group.get_last_center_position()
                    if center_position:
                        cv2.putText(
                            img, str(cnt) + "_" + str(moving_obj_group.get_id()), 
                            (int(center_position[0]), int(center_position[1])), 
                            font_face, font_scale, (0, 255, 0), font_thickness  # SCALAR_GREEN in BGR
                        )
                else:
                    color = (0, 255, 255)  # SCALAR_YELLOW in BGR
    
            cnt += 1


    # Other supporting methods like add_new_moving_object, match_found_obj_to_existing_ones, etc., would go here.
    def add_new_moving_object(self, current_frame_moving_obj):
        moving_obj_group = MovingObjectGroup()
        if not moving_obj_group:
            return
    
        moving_obj_group.add_moving_object(current_frame_moving_obj)
        moving_obj_group.update_state(True)
        self.moving_objects.append(moving_obj_group)
        
        self.detect_bins_in_object_group(moving_obj_group)
    
        # Assuming `bin_detector` is an attribute with `start_detecting` method
        self.bin_detector.start_detecting()
        
    def detect_bins_in_object_group(self, obj_group):
        # Get the cropped image of the object from the second frame
        object_img = obj_group.get_cropped_image(self.second_image_frame)
    
        # Associate the current task ID with this object group for tracking
        self.task_id_to_obj_group[self.task_id] = obj_group
    
        # Load the image as a detection task in the bin detector and increment task ID
        self.bin_detector.load_task(self.task_id, object_img)
        self.task_id += 1

    def get_closest_moving_object(
        self, moving_obj_group_list: List[MovingObjectGroup], target_moving_object: MovingObject
    ) -> Tuple[Optional[MovingObjectGroup], float]:
        closest_moving_obj_group = None
        least_distance = float('inf')

        for moving_obj_group in moving_obj_group_list:
            if moving_obj_group.still_being_tracked():
                last_center_position = moving_obj_group.get_last_center_position()
                if last_center_position is not None:
                    # Calculate distance between the last center position and target's center
                    distance = distance_between_points(last_center_position, target_moving_object.get_center())
                    if distance < least_distance:
                        least_distance = distance
                        closest_moving_obj_group = moving_obj_group

        return closest_moving_obj_group, least_distance
        
    def match_found_obj_to_existing_ones(self, current_frame_moving_objs: List[MovingObject]):
        # Update state of all existing moving object groups to not found
        for moving_obj_group in self.moving_objects:
            moving_obj_group.update_state(False)

        # Match current frame moving objects to existing groups
        for moving_obj in current_frame_moving_objs:
            closest_group, least_distance = self.get_closest_moving_object(self.moving_objects, moving_obj)

            if closest_group:
                closest_group.add_moving_object(moving_obj)
                closest_group.update_state(True)
                
                if  closest_group.bin_id == 0:
                    self.detect_bins_in_object_group(closest_group)
            else:
                self.add_new_moving_object(moving_obj)

        # Remove groups that are not being tracked
        self.moving_objects = [group for group in self.moving_objects if group and group.still_being_tracked()]
    
        # Remove any object groups that are no longer being tracked
        self.moving_objects = [obj_group for obj_group in self.moving_objects if obj_group and obj_group.still_being_tracked()]
    
    def __del__(self):
        print("DELETING OBJECT TRACKER")
