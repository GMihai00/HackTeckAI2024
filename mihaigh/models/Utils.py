
import math
from sklearn.cluster import KMeans
import numpy as np
import cv2

def distance_between_points(point1, point2):
    int_x = abs(point1[0] - point2[0])
    int_y = abs(point1[1] - point2[1])

    return math.sqrt(int_x ** 2 + int_y ** 2)

def get_average_color(image):
    average_color = cv2.mean(image)[:3]  # Ignoring the alpha channel if it exists

    return tuple(map(int, average_color))

def calculate_color_similarity(image, average_color, threshold=50):
    """
    Calculate the percentage of pixels similar to the average color.

    :param image: Input image in BGR format
    :param average_color: Average color of the image
    :param threshold: Color similarity threshold
    :return: Percentage of similar pixels
    """
    # Calculate the difference between each pixel and the average color
    diff = cv2.absdiff(image, average_color)
    
    # Convert to grayscale to simplify comparison
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Create a binary mask where the differences are less than the threshold
    _, binary_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY_INV)

    # Count similar pixels
    similar_pixels_count = cv2.countNonZero(binary_mask)
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate the percentage of similar pixels
    similarity_percentage = (similar_pixels_count / total_pixels) * 100
    return similarity_percentage