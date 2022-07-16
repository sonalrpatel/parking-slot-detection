import math
import cv2
from random import shuffle
from tensorflow import keras

from utils.utils import *
from utils.utils import convert2rgb, preprocess_input


def YoloAnnotationPair(annotation_line):
    annotation_pair = [annotation_line.split()[0],
                       np.array([np.array(list(map(int, box.split(',')))) for box in annotation_line.split()[1:]])]
    return annotation_pair


class YoloDataGenerator(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, do_aug):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.num_classes = num_classes
        self.num_samples = len(self.annotation_lines)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.num_scales = len(self.anchors_mask)
        self.do_aug = do_aug

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """""
        return math.ceil(self.num_samples / float(self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data when the batch corresponding to a given index is called,
        the generator executes the __getitem__ method to generate it.
        """""
        # Generate indexes for a batch
        batch_indexes = range(index * self.batch_size, (index + 1) * self.batch_size)

        # Generate data
        image_data, y_true = self.__data_generation(batch_indexes)

        return [image_data, *y_true], np.zeros(self.batch_size)

    def __data_generation(self, batch_indexes):
        """
        Generates data containing batch_size samples
        """""
        image_data = []
        box_data = []
        for i in batch_indexes:
            i = i % self.num_samples

            annotation_pair = YoloAnnotationPair(self.annotation_lines[i])
            image, box = self.process_data(annotation_pair, self.input_shape, random=self.do_aug)
            image_data.append(preprocess_input(np.array(image)))
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = self.preprocess_true_boxes(box_data)

        return image_data, y_true

    def on_epoch_begin(self):
        """
        Shuffle indexes at start of each epoch
        """""
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def process_data(self, img_bboxes_pair, input_shape, max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5,
                     random=True):
        """
        Random preprocessing for real-time data augmentation
        """""
        # =======================================
        #   Read image and convert to RGB image
        # =======================================
        image = Image.open(img_bboxes_pair[0])
        image = convert2rgb(image)

        # =======================================
        #   Get the height and width of the image and the target height and width
        # =======================================
        iw, ih = image.size
        h, w = input_shape

        # =======================================
        #   Get prediction box
        # =======================================
        box = img_bboxes_pair[1]

        if not random:
            # =======================================
            #   Resize image
            # =======================================
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # =======================================
            #   Add gray bars to the extra parts of the image
            # =======================================
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # =======================================
            #   Adjust the real box
            # =======================================
            box_data = np.zeros((max_boxes, 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                if len(box) > max_boxes:
                    box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data

        # =======================================
        #   Scale the image and distort the length and width
        # =======================================
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # =======================================
        #   Add gray bars to the extra parts of the image
        # =======================================
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # =======================================
        #   Flip the image
        # =======================================
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # =======================================
        #   Gamut distortion
        # =======================================
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

        # =======================================
        #   Adjust the real box
        # =======================================
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            if len(box) > max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box

        return image_data, box_data
    
    def preprocess_true_boxes(self, true_boxes):
        """
        Preprocess true boxes to training input format
        :param
            true_boxes: array, shape=(m, T, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        :return:
            y_true: list of array, shape like yolo_outputs, xywh are relative value
        """""
        assert (true_boxes[..., 4] < self.num_classes).all(), 'class id must be less than num_classes'

        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(self.input_shape, dtype='int32')

        # =======================================
        #   grid_shapes -> [[13,13], [26,26], [52,52]]
        # =======================================
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[s] for s in range(self.num_scales)]

        # =======================================
        #   self.batch_size -> number of images
        #   y_true -> [(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)]
        # =======================================
        # [num_scales][batch_size x (grid_shape_0 x grid_shape_1) x num_anchors_per_scale x (5 + num_classes)]
        y_true = [np.zeros((self.batch_size, grid_shapes[s][0], grid_shapes[s][1], len(self.anchors_mask[s]),
                            5 + self.num_classes), dtype='float32') for s in range(self.num_scales)]

        # =======================================
        #   calculate center point xy, box width and box height
        #   boxes_xy shape -> (m,n,2)  boxes_wh -> (m,n,2)
        #   (x_min, y_min, x_max, y_max) is converted to (x_center, y_center, width, height) relative to input shape
        # =======================================
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

        # =======================================
        #   normalization
        # =======================================
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        # =======================================
        # 	Expand dim to apply broadcasting
        #   [9,2] -> [1,9,2]
        # =======================================
        anchors = np.expand_dims(self.anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes

        # =======================================
        # 	number of non zero boxes
        #   only retrieve image width > 0
        # =======================================
        num_nz_boxes = (np.count_nonzero(boxes_wh, axis=1).sum(axis=1) / 2).astype('int32')

        # =======================================
        #   Loop all the image
        # =======================================
        for b_idx in range(self.batch_size):
            # =======================================
            #   Discard zero rows
            # =======================================
            box_wh = boxes_wh[b_idx, 0:num_nz_boxes[b_idx]]
            if box_wh.shape[0] == 0:
                continue

            # =======================================
            # 	Expand dim to apply broadcasting
            #   [n,2] -> [n,1,2]
            # =======================================			
            box_wh = np.expand_dims(box_wh, -2)
            box_maxes = box_wh / 2.
            box_mins = -box_maxes

            # =======================================
            # 	Find intersection area
            #   Calculate IOU between true box and pre-defined anchors
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            # =======================================			
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

            #   Find box area
            box_area = box_wh[..., 0] * box_wh[..., 1]

            #   Find anchor area
            anchor_area = anchors[..., 0] * anchors[..., 1]

            #   Find iou
            iou_anchors = intersect_area / (box_area + anchor_area - intersect_area)

            # =======================================
            #   Find best anchor for each true box
            # =======================================
            best_anchor_indices = np.argmax(iou_anchors, axis=-1)

            # =======================================
            #   loop all the best anchors, try to find it to which feature layer below
            #   (m 13, 13, 3, 85), (m 26, 26, 3, 85),  (m 52, 52, 3, 85)
            #   y_true shape:
            #       [num_scales][batch_size x (grid_shape_0 x grid_shape_1) x num_anchors_per_scale x (5 + num_classes)]
            # =======================================			
            for box_no, anchor_idx in enumerate(best_anchor_indices):
                # =======================================
                #   Loop all the layers
                # 	[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
                # =======================================
                for s in range(self.num_scales):
                    if anchor_idx in self.anchors_mask[s]:
                        scale_idx = s
                        scale = tuple(grid_shapes[s])

                        # =======================================
                        #   dimensions of a single box
                        # =======================================
                        x, y, width, height = true_boxes[b_idx, box_no, 0:4]

                        # =======================================
                        #   index of the grid cell having the center of the bbox
                        # =======================================
                        i = np.floor(x * scale[1]).astype('int32')
                        j = np.floor(y * scale[0]).astype('int32')

                        # =======================================
                        #   anchor_on_scale -> index of pre-defined anchors
                        # =======================================
                        anchor_on_scale = self.anchors_mask[s].index(anchor_idx)

                        # =======================================
                        #   class_label -> the object category
                        # =======================================
                        class_label = true_boxes[b_idx, box_no, 4].astype('int32')

                        # =======================================
                        #   y_true => shape => (m,13,13,3,85) or (m,26,26,3,85) or (m,52,52,3,85)
                        # =======================================
                        y_true[scale_idx][b_idx, j, i, anchor_on_scale, 0:4] = np.array([x, y, width, height])
                        y_true[scale_idx][b_idx, j, i, anchor_on_scale, 4] = 1
                        y_true[scale_idx][b_idx, j, i, anchor_on_scale, 5 + class_label] = 1

        return y_true
