import cv2

from .MovingObject import MovingObject

class ImageProcessor:
    @staticmethod
    def get_img_convex_hulls(img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        convex_hulls = [cv2.convexHull(contour) for contour in contours]
        return convex_hulls

    @staticmethod
    def is_minimum_obj_size(obj):
        # HARDCODED VALUES TO MANUALLY TWEEKED TO CATCH ONLY BIG ENOUGH OBJECTS TO BE CONTAINERS
        return (obj and obj.get_area() > 4500 and
            0.2 < obj.get_aspect_ratio() < 4.0 and
            obj.get_width() > 130 and obj.get_height() > 130 and
            obj.get_diagonal_size() > 200.0 and
            obj.get_contour_area_ratio() > 0.75)

    def get_moving_objects_from_img(self, img):
        convex_hulls = self.get_img_convex_hulls(img)
        detected_moving_objs = []
        for convex_hull in convex_hulls:
            possible_moving_object = MovingObject(convex_hull)
            if self.is_minimum_obj_size(possible_moving_object):
                detected_moving_objs.append(possible_moving_object)
        return detected_moving_objs

    @staticmethod
    def preprocess_image(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        return image_blurred

    def get_processed_merged_image(self, img1, img2):
        first_processed_image = self.preprocess_image(img1)
        second_processed_image = self.preprocess_image(img2)
        
        img_difference = cv2.absdiff(first_processed_image, second_processed_image)
        _, img_thresh = cv2.threshold(img_difference, 30, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for _ in range(2):
            img_thresh = cv2.dilate(img_thresh, kernel)
            img_thresh = cv2.dilate(img_thresh, kernel)
            img_thresh = cv2.erode(img_thresh, kernel)
        
        return img_thresh
