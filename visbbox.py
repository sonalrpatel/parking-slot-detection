import os
import cv2
import colorsys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description='Visualizing annotation bounding boxes.')
parser.add_argument('image_path', help='Path to the image.')
parser.add_argument('annotation_txt_path', help='Path to the annotation text file.')


def read_lines(path):
    with open(path) as f:
        lines = f.readlines()
    return lines


def draw_boxes(image, annotation):
    num_classes = 4

    # =====================================================================
    #   Picture frame set different colors
    # =====================================================================
    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # =====================================================================
    #   Set font and border thickness
    # =====================================================================
    font = ImageFont.truetype(font='font/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = int(max((image.size[0] + image.size[1]) // np.mean(image.size), 1))

    # =====================================================================
    #   Image drawing
    # =====================================================================
    for e, b in enumerate(annotation):
        cls = b[4]
        box = b[0:4]

        left, top, right, bottom = box

        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))

        label = '{}'.format(cls)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        label = label.encode('utf-8')
        print(label, left, top, right, bottom)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[cls])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[cls])
        draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        del draw
    return image


# %%
def _main(args):
    image = Image.open(args.image_path)

    annotation_lines = read_lines(args.annotation_txt_path)
    annotation_pairs = [[args.annotation_txt_path.rsplit('/', 1)[0] + '/' + line.split()[0],
                         np.array([list(map(int, box.split(','))) for box in line.split()[1:]])]
                        for line in annotation_lines]

    annotation = []
    for pair in annotation_pairs:
        if pair[0] == args.image_path:
            annotation = [list(bbox) for bbox in list(pair[1])]
            break

    print(args.image_path)
    print(annotation)
    draw_boxes(image, annotation)
    image.show()


if __name__ == '__main__':
    # run following command (as per current folder structure) on terminal
    # python visbbox.py data/demo/train/20160725-7-299.jpg data/demo/train_annotations.txt
    # python visbbox.py data/demo/train/20160725-5-652.jpg data/demo/train_annotations.txt
    # python visbbox.py data/demo/train/20160816-1-3003.jpg data/demo/train_annotations.txt
    _main(parser.parse_args())
