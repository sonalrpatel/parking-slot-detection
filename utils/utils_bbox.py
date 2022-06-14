import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# ==============================================================
# adjust with box coordinates to match the original image
# ==============================================================
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # ==============================================================
    # revers y ans h to first dimension
    # ==============================================================
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])

    return boxes


# ==============================================================
#   Adjust predicted result to align with original image
# ==============================================================
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    # ==============================================================
    #   grid_shape = (width, height) = (13, 13) or (26, 26) or (52, 52)
    # ==============================================================
    grid_shape = K.shape(feats)[1:3]

    # ==============================================================
    #   generate grip with shape => (13, 13, num_anchors, 2) => by default (13, 13, 3, 2)
    # ==============================================================
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))

    # ==============================================================
    #   adjust pre-defined anchors to shape (13, 13, num_anchors, 2)
    # ==============================================================
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    # ==============================================================
    #   reshape prediction results to (batch_size,13,13,3,85)
    #   85 = 4 + 1 + 80
    #   4 -> x offset, y offset, width and height
    #   1 -> confidence score
    #   80 -> 80 classes
    # ==============================================================
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ==============================================================
    #   calculate bounding box center point bx, by, width(bw), height(bh) and normalized by grid shape (13, 26 or 52)
    #   bx = sigmoid(tx) + cx
    #   by = sigmoid(tx) + cy
    #   bw = pw * exp(tw)
    #   bh = ph * exp(th)
    # ==============================================================
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    # ==============================================================
    #   retrieve confidence score and class probs
    # ==============================================================
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # ==============================================================
    #   if calc loss return -> grid, feats, box_xy, box_wh
    #   if during prediction return -> box_xy, box_wh, box_confidence, box_class_probs
    # ==============================================================
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


# ==============================================================
#   Adjust predicted result to align with original image
#   Convert final layer features to bounding box parameters
# ==============================================================
def get_pred_boxes(final_layer_feats, anchors, num_classes, input_shape, calc_loss=False):
    # ==============================================================
    #   number of anchors
    # ==============================================================
    num_anchors = len(anchors)

    # ==============================================================
    #   (m is batch_size)
    #   final layer feature shape is (m, 13, 13, 255) or (m, 26, 26, 255) or (m, 52, 52, 255) as per scales
    #   grid_shape = (width, height) = (13, 13) or (26, 26) or (52, 52)
    # ==============================================================
    grid_shape = K.shape(final_layer_feats)[1:3]

    # ==============================================================
    #   generate grid with shape => (grid_shape[0], grid_shape[1], num_anchors, "2" for x & y) => e.g. (13, 13, 3, 2)
    # ==============================================================
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(final_layer_feats))

    # ==============================================================
    #   adjust pre-defined anchors to shape (13, 13, num_anchors, 2)
    # ==============================================================
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    # ==============================================================
    #   reshape prediction results (m, 13, 13, 255) to (m, 13, 13, 3, 85)
    #   85 = 4 + 1 + 80
    #   4 -> x offset, y offset, width and height
    #   1 -> confidence score
    #   80 -> 80 classes
    # ==============================================================
    final_layer_feats = K.reshape(final_layer_feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ==============================================================
    #   calculate bounding box center point bx, by and normalize by grid shape (13, 26 or 52)
    #   bx = sigmoid(tx) + cx
    #   by = sigmoid(tx) + cy
    #   calculate bounding box width(bw), height(bh) and normalize by input shape (416)
    #   bw = pw * exp(tw)
    #   bh = ph * exp(th)
    # ==============================================================
    box_xy = (K.sigmoid(final_layer_feats[..., :2]) + grid) / K.cast(grid_shape, K.dtype(final_layer_feats))
    box_wh = K.exp(final_layer_feats[..., 2:4]) * anchors_tensor / K.cast(input_shape, K.dtype(final_layer_feats))

    # ==============================================================
    #   concat predicted xy and wh to bounding box shape => (m,13,13,3,4)
    # ==============================================================
    pred_box = K.concatenate([box_xy, box_wh])

    # ==============================================================
    #   if calc loss, then return -> grid, feats, box_xy, box_wh
    #   if during prediction, then return -> box_xy, box_wh, box_confidence, box_class_probs
    # ==============================================================
    if calc_loss:
        ret = [grid, final_layer_feats, pred_box]
    else:
        # ==============================================================
        #   retrieve confidence score and class probabilities
        # ==============================================================
        box_confidence = K.sigmoid(final_layer_feats[..., 4:5])
        box_class_probs = K.sigmoid(final_layer_feats[..., 5:])

        ret = [box_xy, box_wh, box_confidence, box_class_probs]

    return ret


# ==============================================================
#   Decode model outputs, Correct sizes, Non-max suppression and Return
#   1 - box coordinates (x1, y1, x2, y2)
#   2 - confidence score
#   3 - classes score
# ==============================================================
def DecodeBox(outputs,  # raw outputs from YoloV3
              anchors,  # pre-defined anchors in configuration
              num_classes,  # COCO=80, VOC=20
              input_shape,  # image shape 416 * 416
              # ==============================================================
              #   13x13's anchor are [116,90],[156,198],[373,326]
              #   26x26's anchors are [30,61],[62,45],[59,119]
              #   52x52's anchors are [10,13],[16,30],[33,23]
              # ==============================================================
              anchor_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
              max_boxes=100,
              conf_thresh=0.5,
              nms_iou_thresh=0.3,
              letterbox_image=True):
    # ==============================================================
    #   Decode the output of the model/network
    # ==============================================================
    box_xy = []
    box_wh = []
    box_confidence = []
    box_class_probs = []
    # loop for number of layers/scales/shapes (3 in yolov3) in final layer features
    for i in range(len(anchor_mask)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_pred_boxes(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))

    box_xy = K.concatenate(box_xy, axis=0)
    box_wh = K.concatenate(box_wh, axis=0)
    box_confidence = K.concatenate(box_confidence, axis=0)
    box_class_probs = K.concatenate(box_class_probs, axis=0)

    # ==============================================================
    #   Before image pass into Yolo network there is a pre-process method letter_box_image will padding gray points
    #       around image if size is not enough.
    #   So predicted box_xy, box_wh need to be adjusted to align with previous image and convert to Xmin, Ymin and
    #       Xmax, Ymax format.
    #   If model skip letterbox_image pre-process method, here still need to scale up to align with
    #       original image due to normalization.
    # ==============================================================
    input_image_shape = K.reshape(outputs[-1], [-1])
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, input_image_shape, letterbox_image)

    # ==============================================================
    #   Consider boxes having score lesser than score threshold
    #   Perform non-max suppression
    # ==============================================================
    box_scores = box_confidence * box_class_probs
    mask = (box_scores >= conf_thresh)
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out = []
    scores_out = []
    classes_out = []
    for c in range(num_classes):
        # ==============================================================
        #   retrieve all the boxes and box scores >= score threshold
        # ==============================================================
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # ==============================================================
        #   retrieve NMS index via IOU threshold
        # ==============================================================
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                 iou_threshold=nms_iou_thresh)

        # ==============================================================
        #   retrieve boxes, boxes scores and classes via NMS index
        # ==============================================================
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)

    boxes_out = K.concatenate(boxes_out, axis=0)
    scores_out = K.concatenate(scores_out, axis=0)
    classes_out = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


if __name__ == "__main__":
    outputs = tf.random.normal([2, 13, 13, 255])
    anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = (416, 416)

    grid, raw_pred, pred_xy, pred_wh = get_pred_boxes(outputs, anchors[anchors_mask[0]], 80, input_shape)

    print(input_shape.__len__(), outputs[-1].shape)
