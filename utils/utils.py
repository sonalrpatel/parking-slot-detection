import os
import numpy as np
from PIL import Image
from functools import reduce


def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


# ===================================================================
#   Convert image to RGB
#   Currently only support RGB, all the image should convert to RGB before send into model
# ===================================================================
def convert2rgb(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, letterbox_image):
    in_w, in_h = image.size
    net_w, net_h = size
    if letterbox_image:
        scale = min(net_w / in_w, net_h / in_h)
        int_w = int(in_w * scale)
        int_h = int(in_h * scale)

        image = image.resize((int_w, int_h), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((net_w - int_w) // 2, (net_h - int_h) // 2))
    else:
        new_image = image.resize((net_w, net_h), Image.BICUBIC)
    return new_image


def get_classes(class_file_name):
    """
    loads class name from a file
    :param class_file_name: 
    :return: class names 
    """""
    if not os.path.exists(class_file_name):
        class_file_name = os.path.join(os.path.dirname(__file__), '..', class_file_name)

    class_names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')
    return class_names, len(class_names)


def get_anchors(anchors_path):
    """
    loads the anchors from a file
    """""
    if not os.path.exists(anchors_path):
        anchors_path = os.path.join(os.path.dirname(__file__), '..', anchors_path)

    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)


def preprocess_input(image):
    image /= 255.0
    return image
