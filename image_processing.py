import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.morphology import binary_closing
from skimage import measure
from skimage.feature import canny
from skimage.measure import regionprops
from scipy import ndimage as ndi


def reduce_image(image, reduce_coef):
    image_width = image.shape[1] * reduce_coef / 100
    image_height = image.shape[0] * reduce_coef / 100
    return cv2.resize(image, (int(image_width), int(image_height)))


def get_components(mask):
    labels = measure.label(mask)  # разбиение маски на компоненты связности
    props = regionprops(labels)  # нахождение свойств каждой области

    return labels, props


def get_largest_component(mask):
    labels, props = get_components(mask)
    areas = [prop.area for prop in props]  # нас интересуют площади компонент связности

    largest_comp_id = np.array(areas).argmax()  # находим номер компоненты с максимальной площадью

    return labels == (largest_comp_id + 1)


def cut_and_get_polygon(image):
    img_gray = rgb2gray(image)
    gaussian_sigma = 1.5
    img_gray_blur = gaussian(img_gray, sigma=gaussian_sigma, channel_axis=True)

    canny_sigma = 1.5
    canny_low_threshold = 0.2
    canny_high_threshold = 0.6
    res_image = canny(img_gray_blur, sigma=canny_sigma, low_threshold=canny_low_threshold,
                      high_threshold=canny_high_threshold)

    res_image = binary_closing(res_image, footprint=np.ones((5, 5)))

    res_image = ndi.binary_fill_holes(res_image)

    polygon = res_image[0:(res_image.shape[0] // 2), 0:res_image.shape[1]]

    polygon = get_largest_component(polygon)

    return polygon


def correct_mask_borders_after_canny(canny_result, border_width=3):
    canny_result[:border_width, :] = 0
    canny_result[:, :border_width] = 0
    canny_result[-border_width:, :] = 0
    canny_result[:, -border_width:] = 0


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


def cut_box(mask):
    mask_uint8 = mask.astype(np.uint8).copy()
    x, y, w, h = cv2.boundingRect(mask_uint8)

    box = [[0] * w for i in range(h)]

    for i in range(x, x + w):
        for j in range(y, y + h):
            box[j - y][i - x] = mask_uint8[j][i]

    return box


def calc_area(mask):
    area = 0
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if (mask[i][j]) == 1:
                area += 1

    return area


def extract_polygon(image):
    polygon = cut_and_get_polygon(image)
    polygon = cut_box(polygon)
    return polygon


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


def compare_ares(polygon, items):
    polygon_area = calc_area(polygon)
    items_area = sum(calc_area(item) for item in items)
    if items_area < polygon_area:
        return True
    else:
        return False


def check_for_place(folder_path):
    for image_path in os.listdir(folder_path):
        image_file = os.path.join(folder_path, image_path)
        image = imread(image_file)
        reducing_image = reduce_image(image, 50)

        polygon = extract_polygon(reducing_image)
        fig1, ax1 = plt.subplots()
        ax1.imshow(polygon)

        foot_width = 10
        if image_path == 'only_one_object_in.jpg':
            foot_width = 6

        items = extract_items(reducing_image, foot_width)
        i = 0
        fig2, ax2 = plt.subplots(1, len(items))
        for item in items:
            if len(items) == 1:
                ax2.imshow(item)
                continue
            ax2[i].imshow(item)
            i += 1

        plt.show()
        print(compare_ares(polygon, items))


# check_for_place("dataset")
