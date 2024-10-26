from typing import List, Optional, Tuple
import cv2
import numpy as np
import threading

class MovingObjectGroup:
    MAX_OBJECTS_STORED = 10
    MAX_FRAMES_WITHOUT_A_MATCH = 5
    
    instance_count = 0
    
    def __init__(self):
        self.center_positions = []
        self.future_position = (0, 0)
        self.moving_object_states = []
        self.sum_center_pos = (0, 0)
        self.obj_found_in_frame = False
        self.nr_frames_without_match = 0
        self.nr_frames_without_being_bin = 0
        self.nr_bins = 0
        self.mutex_group = threading.Lock()
        
        MovingObjectGroup.instance_count += 1
        self.id = MovingObjectGroup.instance_count
    
    
    def get_id(self):
        return self.id
        
    def predict_next_position(self):
        n = len(self.center_positions)
        if n == 0:
            return
        elif n == 1:
            self.future_position = self.center_positions[-1]
        else:
            delta_x = (self.center_positions[-1][0] * (n - 1) - self.sum_center_pos[0]) / ((n * (n - 1)) / 2.0)
            delta_y = (self.center_positions[-1][1] * (n - 1) - self.sum_center_pos[1]) / ((n * (n - 1)) / 2.0)
            self.future_position = (int(self.center_positions[-1][0] + delta_x),
                                    int(self.center_positions[-1][1] + delta_y))

    def add_moving_object(self, obj):
        with self.mutex_group:
            if len(self.moving_object_states) == self.MAX_OBJECTS_STORED:
                old_pos = self.center_positions.pop(0)
                self.sum_center_pos = (self.sum_center_pos[0] - old_pos[0], self.sum_center_pos[1] - old_pos[1])
                self.moving_object_states.pop(0)

            self.moving_object_states.append(obj)
            center = obj.get_center()
            self.center_positions.append(center)

            if len(self.moving_object_states) > 1:
                self.sum_center_pos = (self.sum_center_pos[0] + center[0], self.sum_center_pos[1] + center[1])

            self.predict_next_position()

    def update_state(self, found: bool):
        with self.mutex_group:
            if found:
                self.obj_found_in_frame = True
                self.nr_frames_without_match = 0
            else:
                self.nr_frames_without_match += 1
                self.obj_found_in_frame = False

    def get_nr_of_moving_objects_in_group(self) -> int:
        with self.mutex_group:
            return len(self.center_positions)

    def still_being_tracked(self) -> bool:
        with self.mutex_group:
            return (self.nr_frames_without_match < self.MAX_FRAMES_WITHOUT_A_MATCH and
                    self.nr_frames_without_being_bin < self.MAX_FRAMES_WITHOUT_A_MATCH)

    def get_center_position(self, index: int) -> Optional[Tuple[int, int]]:
        with self.mutex_group:
            if index >= len(self.center_positions):
                return None
            return self.center_positions[index]

    def get_last_center_position(self) -> Optional[Tuple[int, int]]:
        with self.mutex_group:
            return self.center_positions[-1] if self.center_positions else None

    def get_future_position(self) -> Tuple[int, int]:
        with self.mutex_group:
            return self.future_position

    def get_diagonal_size(self) -> Optional[float]:
        with self.mutex_group:
            if not self.moving_object_states:
                return None
            return self.moving_object_states[0].get_diagonal_size()

    def get_last_state(self):
        with self.mutex_group:
            return self.moving_object_states[-1] if self.moving_object_states else None

    def get_first_state(self):
        with self.mutex_group:
            return self.moving_object_states[0] if self.moving_object_states else None

    def get_cropped_image(self, img: np.ndarray) -> np.ndarray:
        with self.mutex_group:
            if not self.moving_object_states:
                return img
    
            try:
                # Convert rect to a list to allow modifications
                rect = list(self.moving_object_states[-1].get_bounding_rect())
                rect[2] = int(rect[2] * 1.5)  # Modify width
                rect[3] = int(rect[3] * 1.5)  # Modify height
                rect[0] -= 20  # Adjust x-coordinate
                rect[1] -= 20  # Adjust y-coordinate
                
                # Ensure that the cropping region is within the image boundaries
                rect[0] = max(0, rect[0])
                rect[1] = max(0, rect[1])
                rect[2] = min(rect[2], img.shape[1] - rect[0])
                rect[3] = min(rect[3], img.shape[0] - rect[1])
                
                return img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            except Exception as e:
                # If there's an error, return the original bounding rectangle of the last object state
                return img[self.moving_object_states[-1].get_bounding_rect()]

    def update_bin_state(self, nr_bins: int):
        with self.mutex_group:
            if nr_bins == 0:
                self.nr_frames_without_being_bin += 1
            else:
                self.nr_frames_without_being_bin = 0
            self.nr_bins = nr_bins

    def get_nr_bins(self) -> int:
        with self.mutex_group:
            return self.nr_bins
