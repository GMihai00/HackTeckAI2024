import cv2
import numpy as np

class MovingObject:
    def __init__(self, contour):
        self.contour = contour
        self.bounding_rect = cv2.boundingRect(self.contour)

    def get_center(self):
        center_x = (self.bounding_rect[0] + self.bounding_rect[0] + self.bounding_rect[2]) // 2
        center_y = (self.bounding_rect[1] + self.bounding_rect[1] + self.bounding_rect[3]) // 2
        return (center_x, center_y)

    def get_bbox(self):
        return self.bounding_rect
        
    def get_diagonal_size(self):
        width = self.bounding_rect[2]
        height = self.bounding_rect[3]
        return np.sqrt(width ** 2 + height ** 2)

    def get_aspect_ratio(self):
        width = self.bounding_rect[2]
        height = self.bounding_rect[3]
        return float(width) / height if height != 0 else 0

    def get_area(self):
        width = self.bounding_rect[2]
        height = self.bounding_rect[3]
        return width * height

    def get_contour_area_ratio(self):
        bounding_area = self.get_area()
        contour_area = cv2.contourArea(self.contour)
        return contour_area / bounding_area if bounding_area != 0 else 0

    def get_width(self):
        return self.bounding_rect[2]

    def get_height(self):
        return self.bounding_rect[3]

    def get_contour(self):
        return self.contour

    def get_bounding_rect(self):
        return self.bounding_rect
