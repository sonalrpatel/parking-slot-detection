import tensorflow as tf
from tensorflow.keras import backend as K
from utils.utils_bbox import get_anchors_and_decode


# ---------------------------------------------------#
#   calculate iou between predicted boxes and true boxes
# ---------------------------------------------------#
def box_iou(b1, b2):
    # ---------------------------------------------------#
    #   num_anchor,1,4
    #   calculate top left and bottom right coordinates
    # ---------------------------------------------------#
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # ---------------------------------------------------#
    #   1,n,4
    #   calculate top left and bottom right coordinates
    # ---------------------------------------------------#
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # ---------------------------------------------------#
    #   calculate IOU
    # ---------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# ---------------------------------------------------#
#   loss function
# ---------------------------------------------------#
def yolo_loss(args, input_shape, anchors, anchors_mask, num_classes, loss_iou_thresh):
    num_layers = len(anchors_mask)
    # ---------------------------------------------------------------------------------------------------#
    #   split predictions and ground truth, args is list contains [*model_body.output, *y_true]
    #   y_true is a list，contains 3 feature maps，shape are:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #   yolo_outputs is a list，contains 3 feature maps，shape are:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    # ---------------------------------------------------------------------------------------------------#
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # -----------------------------------------------------------#
    #   input_shpae = (416, 416)
    # -----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))
    # -----------------------------------------------------------#
    #   grid shapes  = [[13,13], [26,26], [52,52]]
    # -----------------------------------------------------------#
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    # -----------------------------------------------------------#
    #   m = batch_size = number of images
    # -----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]

    loss = 0
    num_pos = 0
    # ---------------------------------------------------------------------------------------------------#
    #   y_true is a list，contains 3 feature maps，shape are: (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    #   yolo_outputs is a list，contains 3 feature maps，shape are:(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    # ---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        # -----------------------------------------------------------#
        #   Here take fist feature map as example (m,13,13,3,85)
        #   Retrieve object score from last dim with shape => (m,13,13,3,1)
        # -----------------------------------------------------------#
        object_mask = y_true[l][..., 4:5]
        # -----------------------------------------------------------#
        #   Retrieve object category from last dim with shape => (m,13,13,3,80)
        # -----------------------------------------------------------#
        true_class_probs = y_true[l][..., 5:]

        # -----------------------------------------------------------#
        #   decode Yolo predictions will return below 4 matrix as below:
        #   grid        (13,13,1,2) grid coordinates
        #   raw_pred    (m,13,13,3,85) raw prediction
        #   pred_xy     (m,13,13,3,2) decode box center point x and y
        #   pred_wh     (m,13,13,3,2) decode width and height
        # -----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
                                                                  anchors[anchors_mask[l]], num_classes, input_shape,
                                                                  calc_loss=True)

        # -----------------------------------------------------------#
        #   concat predicted xy and wh to shape => (m,13,13,3,4)
        # -----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        # -----------------------------------------------------------#
        #   create a dynamic array to save negative samples
        # -----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        # -----------------------------------------------------------#
        #   define a inner function to locate ignored samples
        # -----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            # -----------------------------------------------------------#
            #  retrieve ground truth's bounding box's coordinates (x,y) and (w,h)=> shape (n, 4)
            # -----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # -----------------------------------------------------------#
            #   calculate iou
            #   pred_box shape => (13,13,3,4)
            #   true_box shape => (n,4)
            #   iou between predicted box and true box, shape => (13,13,3,n)
            # -----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            # -----------------------------------------------------------#
            #   best_iou shape => (13,13,3) means best iou for every anchor
            # -----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            # -----------------------------------------------------------#
            #   if best iou less than ignore threshold treat it as ignored sample
            #   add to ignore mask for calculate no object loss latter
            #   if best iou is greater or equal than ignore threshold, we think it's
            #   close to the true box will not treat it as a ignored sample
            # -----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < loss_iou_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        # -----------------------------------------------------------#
        #   call while loop to find out ignored samples in every images one by one
        # -----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])

        # -----------------------------------------------------------#
        #   ignore_mask shape => (m,13,13,3)
        # -----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #  (m,13,13,3) =>  (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # -----------------------------------------------------------#
        #   normalize true xy and wh to align with predicted xy and wh for calculate loss latter
        # -----------------------------------------------------------#
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchors_mask[l]] * input_shape[::-1])

        # -----------------------------------------------------------#
        #   update raw_true_wh if no object update wh to 0 otherwise remain it as before
        # -----------------------------------------------------------#
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
        # -----------------------------------------------------------#
        #  calculate box loss scale, x and y both between 0-1
        #  if real box is bigger box_loss_scale will become smaller
        #  if real box is smaller box_loss_scale will become larger
        #  to make sure big box and smaller box to have almost same loss value
        # -----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # -----------------------------------------------------------#
        #   use binary_crossentropy calculate xy loss
        # -----------------------------------------------------------#
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        # -----------------------------------------------------------#
        #   calculate wh_loss
        # -----------------------------------------------------------#
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])

        # ------------------------------------------------------------------------------#
        #   if there is true box，calculate cross entropy loss between predicted score and 1
        #   if there no box，calculate cross entropy loss between predicted score and 0
        #   and will ignore the samples if best_iou<loss_iou_thresh
        # ------------------------------------------------------------------------------#
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        # -----------------------------------------------------------#
        #   sum up loss
        # -----------------------------------------------------------#
        xy_loss = K.sum(xy_loss)
        wh_loss = K.sum(wh_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)
        # -----------------------------------------------------------#
        #   add up all the loss
        # -----------------------------------------------------------#
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss += xy_loss + wh_loss + confidence_loss + class_loss

    loss = loss / num_pos
    return loss
