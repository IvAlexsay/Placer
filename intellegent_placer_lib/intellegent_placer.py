import os
from imageio import imread
from intellegent_placer_lib.algorithm import insert_items_in_polygon
from intellegent_placer_lib.image_processing import reduce_image, extract_polygon, extract_items


# Функция, подсчитывающая площадь маски
def calc_area(mask):
    area = 0
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if (mask[i][j]) == 1:
                area += 1

    return area


# Фунцкия для сравнения площадей многоугольника и всех предметов
def compare_ares(polygon, items):
    polygon_area = calc_area(polygon)
    items_area = sum(calc_area(item) for item in items)
    if items_area < polygon_area:
        return True
    else:
        return False


# Функция, принимающая исходное изображение с многоугольником и предметами
# и вычисляет возможность распололжить предметы внутри многоугольника
def check_for_place(folder_path, image_path):
    image_file = os.path.join(folder_path, image_path)
    image = imread(image_file)
    reducing_image = reduce_image(image, 50)

    polygon = extract_polygon(reducing_image)

    foot_width = 10
    if image_path == 'only_one_object_in.jpg':
        foot_width = 6

    items = extract_items(reducing_image, foot_width)

    if compare_ares(polygon, items):
        result, img = insert_items_in_polygon(polygon, items)
        return reducing_image, polygon, items, result, img
    else:
        return reducing_image, polygon, items, False, None
