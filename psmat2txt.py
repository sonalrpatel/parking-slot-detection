import os
import glob
import scipy.io
import argparse
from PIL import Image
from tqdm import tqdm

WIDTH_MARK = 50
HEIGHT_MARK = 50
deltaWIDTH_HEAD = 40
deltaHEIGHT_HEAD = 40

parser = argparse.ArgumentParser(description='Converting the ps labels from mat into a single annotation text file.')
parser.add_argument('data_path', help='Path to the folder containing the images.')
parser.add_argument('annotation_txt_path', help='Path to the folder to store the generated annotation text.')


def mark2bbox(mark):
    mark_bbox = [0] * 5
    # x-min
    mark_bbox[0] = int(mark[0] - (WIDTH_MARK / 2))
    # y-min
    mark_bbox[1] = int(mark[1] - (HEIGHT_MARK / 2))
    # x-max
    mark_bbox[2] = int(mark[0] + (WIDTH_MARK / 2))
    # y-max
    mark_bbox[3] = int(mark[1] + (HEIGHT_MARK / 2))
    # class = 0 for T-shape or L-shape
    mark_bbox[4] = 0

    return mark_bbox


def head2bbox(slot, marks):
    # x-center = (p1(x) + p2(x)) / 2
    x_center = (marks[slot[0] - 1][0] + marks[slot[1] - 1][0]) / 2
    # y-center = (p1(y) + p2(y)) / 2
    y_center = (marks[slot[0] - 1][1] + marks[slot[1] - 1][1]) / 2

    # width-head = |p1(x) - p2(x)| / 2 + deltaWIDTH_HEAD
    width_head = int(abs(marks[slot[0] - 1][0] - marks[slot[1] - 1][0]) / 2) + deltaWIDTH_HEAD
    # height-head = |p1(y) - p2(y)| / 2 + deltaHEIGHT_HEAD
    height_head = int(abs(marks[slot[0] - 1][1] - marks[slot[1] - 1][1]) / 2) + deltaHEIGHT_HEAD

    head_bbox = [0] * 5
    # x-min
    head_bbox[0] = int(x_center - width_head / 2)
    # y-min
    head_bbox[1] = int(y_center - height_head / 2)
    # x-max
    head_bbox[2] = int(x_center + width_head / 2)
    # y-max
    head_bbox[3] = int(y_center + height_head / 2)

    # class = 1 for right-angle head, 2 for acute-angle and 3 for obtuse-angle
    if slot[3] > 90:
        head_bbox[4] = 3
    elif slot[3] < 90:
        head_bbox[4] = 2
    else:
        head_bbox[4] = 1

    return head_bbox


def head2bbox2(slot, marks):
    p1x, p1y = marks[slot[0] - 1]
    p2x, p2y = marks[slot[1] - 1]

    # Right and Left points identification
    if abs(p1x - p2x) > 0.1:
        if p1x > p2x:
            pRx, pRy = p1x, p1y
            pLx, pLy = p2x, p2y
        if p1x < p2x:
            pRx, pRy = p2x, p2y
            pLx, pLy = p1x, p1y
    else:
        if p1y < p2y:
            pRx, pRy = p1x, p1y
            pLx, pLy = p2x, p2y
        else:
            pRx, pRy = p2x, p2y
            pLx, pLy = p1x, p1y

    # Slope calculation
    if abs(pRx - pLx) > 0.1:
        slope = (pRy - pLy) / (pRx - pLx)
    else:
        slope = 0

    # Bounding box calculation
    if slope > 0:
        pRRx = pRx + deltaWIDTH_HEAD
        pRRy = pRy + deltaHEIGHT_HEAD
        pLLx = (pLx - deltaWIDTH_HEAD) if (pLx - deltaWIDTH_HEAD) > 0 else 0
        pLLy = (pLy - deltaHEIGHT_HEAD) if (pLy - deltaHEIGHT_HEAD) > 0 else 0
    else:  # slope < 0 or == 0
        pRRx = pRx + deltaWIDTH_HEAD
        pRRy = (pRy - deltaHEIGHT_HEAD) if (pRy - deltaHEIGHT_HEAD) > 0 else 0
        pLLx = (pLx - deltaWIDTH_HEAD) if (pLx - deltaWIDTH_HEAD) > 0 else 0
        pLLy = pLy + deltaHEIGHT_HEAD

    head_bbox = [0] * 5
    # x-min
    head_bbox[0] = int(min(pRRx, pLLx))
    # y-min
    head_bbox[1] = int(min(pRRy, pLLy))
    # x-max
    head_bbox[2] = int(max(pRRx, pLLx))
    # y-max
    head_bbox[3] = int(max(pRRy, pLLy))

    # class = 1 for right-angle head, 2 for acute-angle and 3 for obtuse-angle
    if slot[3] > 90:
        head_bbox[4] = 3
    elif slot[3] < 90:
        head_bbox[4] = 2
    else:
        head_bbox[4] = 1

    return head_bbox


# %%
def _main(args):
    images = glob.glob(args.data_path + "*.jpg")
    annotation_lines = []
    for image in tqdm(images):
        matfile = image.rsplit(".", 1)[0] + ".mat"
        mat = scipy.io.loadmat(matfile)

        label = []
        for mark in mat['marks']:
            label.append(mark2bbox(mark))
        for slot in mat['slots']:
            label.append(head2bbox2(slot, mat['marks']))

        # <file path> <x-min> <y-min> <x-max> <y-max> <object class>
        label_str = " ".join([str(l).replace("[", "").replace("]", "").replace(" ", "") for l in label])
        annotation_line = " ".join([image.split("\\")[1], label_str])

        annotation_lines.append(annotation_line)

    with open(args.annotation_txt_path, 'a') as f:
        for line in tqdm(annotation_lines):
            f.write(line)
            f.write('\n')

    print("success")


if __name__ == '__main__':
    # run following command (as per current folder structure) on terminal
    # python psmat2txt.py D:/01_PythonAIML/00_Datasets/ps2.0/training/ data/ps/train_annotations.txt
    _main(parser.parse_args())
