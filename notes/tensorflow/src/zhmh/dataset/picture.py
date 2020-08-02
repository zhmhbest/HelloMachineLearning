import numpy as np


def generate_random_one_rgb_picture(width, height):
    """
    模拟一张RGB图片
    :param width: 宽度
    :param height: 高度
    :return:
    """
    def __point__():
        import random
        return random.randint(0, 255)
    buffer = []
    for i in range(width):
        line = []
        for j in range(height):
            line.append([__point__(), __point__(), __point__()])
        buffer.append(line)
    return buffer


def generate_random_rgb_pictures(width, height, number):
    buffer = []
    for i in range(number):
        buffer.append(generate_random_one_rgb_picture(width, height))
    return np.array(buffer, np.float)


def show_rgb_picture(data):
    # pip install Pillow
    from PIL import Image
    width = len(data)
    height = len(data[0])
    im = Image.new("RGB", (width, height))  # 创建图片
    for i in range(width):
        for j in range(height):
            item = data[i][j]
            im.putpixel((i, j), (int(item[0]), int(item[1]), int(item[2])))
    im.show()