import numpy as np
import cv2
from scipy.ndimage import rotate

ACCEPTABLE_BOUNDS_ERROR = 50
DOTS_STEP = 10
ANGLE_STEP = 5


# Функция, инвертирующая все значения матрицы
def invert_matrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0


# Функция, пытающаяся расположить все предметы внутри многоугольника
def insert_items_in_polygon(polygon, items):
    invert_matrix(polygon)
    arr_polygon = np.array(polygon)

    for item in items:
        arr_item = np.array(item)
        if not insert_item_in_polygon(arr_polygon, arr_item):
            return False, None
        elif len(items) <= 1:
            return True, arr_polygon

    return True, arr_polygon


# Функция, пытающаяся расположить один предмет внутри многоугольника
def insert_item_in_polygon(polygon_mask, object_mask):
    object_mask_height, object_mask_width = object_mask.shape
    polygon_mask_height, polygon_mask_width = polygon_mask.shape

    for y in range(0, polygon_mask_height - object_mask_height, DOTS_STEP):
        for x in range(0, polygon_mask_width - object_mask_width, DOTS_STEP):
            for angle in range(0, 360, ANGLE_STEP):
                rotated_object_mask = rotate(object_mask, angle, reshape=True)
                rotated_object_mask_height, rotated_object_mask_width = rotated_object_mask.shape
                polygon_mask_cut = polygon_mask[y:y + rotated_object_mask_height,
                                   x:x + rotated_object_mask_width]

                try:
                    excess_area = cv2.bitwise_and(polygon_mask_cut.astype(int), rotated_object_mask.astype(int))
                except:
                    continue

                if np.sum(excess_area) < ACCEPTABLE_BOUNDS_ERROR:
                    polygon_mask[y:y + rotated_object_mask_height, x:x + rotated_object_mask_width] = \
                        cv2.bitwise_xor(polygon_mask_cut.astype(int), rotated_object_mask.astype(int)).astype(bool)
                    return True

    return False
