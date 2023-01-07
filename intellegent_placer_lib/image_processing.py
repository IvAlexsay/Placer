import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.morphology import binary_closing
from skimage import measure
from skimage.feature import canny
from skimage.measure import regionprops
from scipy import ndimage as ndi


# Функция уменьшающая разрешение изображения
def reduce_image(image, reduce_coef):
    image_width = image.shape[1] * reduce_coef / 100
    image_height = image.shape[0] * reduce_coef / 100
    return cv2.resize(image, (int(image_width), int(image_height)))


# Фунцкция разбивающая общую маску изображеня на несколько компонент связности,
# и находящая свойства получившихся компонент
def get_components(mask):
    labels = measure.label(mask)
    props = regionprops(labels)

    return labels, props


# Функция, извлекающая наибольший объект из маски
def get_largest_component(mask):
    labels, props = get_components(mask)
    areas = [prop.area for prop in props]  # нас интересуют площади компонент связности

    largest_comp_id = np.array(areas).argmax()  # находим номер компоненты с максимальной площадью

    return labels == (largest_comp_id + 1)


# Функция, извлекающая маску многоугольника их исходного изображения
def cut_and_get_polygon(image):
    img_gray = rgb2gray(image)
    gaussian_sigma = 1.5
    img_gray_blur = gaussian(img_gray, sigma=gaussian_sigma, channel_axis=True)

    canny_sigma = 1.5
    canny_low_threshold = 0.2
    canny_high_threshold = 0.6
    res_image = canny(img_gray_blur, sigma=canny_sigma, low_threshold=canny_low_threshold,
                      high_threshold=canny_high_threshold)

    res_image = binary_closing(res_image, footprint=np.ones((4, 4)))

    res_image = ndi.binary_fill_holes(res_image)

    polygon = res_image[0:(res_image.shape[0] // 2), 0:res_image.shape[1]]

    polygon = get_largest_component(polygon)

    return polygon


# Функция, убирающая границы по краям, найденные фильтром canny
def correct_mask_borders_after_canny(canny_result, border_width=3):
    canny_result[:border_width, :] = 0
    canny_result[:, :border_width] = 0
    canny_result[-border_width:, :] = 0
    canny_result[:, -border_width:] = 0


# Функция, извлекающая предметы из исходного изображения
def cut_and_get_items(image, foot_width):
    img_gray = rgb2gray(image)
    gaussian_sigma = 1.5
    img_gray_blur = gaussian(img_gray, sigma=gaussian_sigma, channel_axis=True)

    canny_sigma = 1.5
    canny_low_threshold = 0.05
    canny_high_threshold = 0.25
    res_image = canny(img_gray_blur, sigma=canny_sigma, low_threshold=canny_low_threshold,
                      high_threshold=canny_high_threshold)

    res_image = binary_closing(res_image, footprint=np.ones((foot_width, foot_width)))

    correct_mask_borders_after_canny(res_image)

    res_image = ndi.binary_fill_holes(res_image)

    items = res_image[(res_image.shape[0] // 2 + 12):res_image.shape[0], 0:res_image.shape[1]]

    return items


# Функция, обрезающая маску по границам объекта
def cut_box(mask):
    mask_uint8 = mask.astype(np.uint8).copy()
    x, y, w, h = cv2.boundingRect(mask_uint8)

    box = [[0] * w for i in range(h)]

    for i in range(x, x + w):
        for j in range(y, y + h):
            box[j - y][i - x] = mask_uint8[j][i]

    return box


# Функция, возвращающая обрезанную маску извлеченного многоугольник
def extract_polygon(image):
    polygon = cut_and_get_polygon(image)
    polygon = cut_box(polygon)
    return polygon


# Функция, возвращающая обрезанную маски извлеченных предметов
def extract_items(image, foot_width):
    combine_items = cut_and_get_items(image, foot_width)
    components, props = get_components(combine_items)
    areas = [(r.area, r.label) for r in props]
    areas.sort(reverse=True)
    areas = [c for c in areas if c[0] > 200]

    items = []
    for a in areas:
        items.append(cut_box(components == a[1]))

    return items
