from tensorflow.keras.layers import Concatenate, Input, UpSampling2D
from tensorflow.keras.models import Model
from utils.utils import compose

from model_yolo3_tf2.darknet import DarknetConv2D, DarknetConv2D_BN_Leaky, darknet_body


# ---------------------------------------------------#
#   Conv * 5
# ---------------------------------------------------#
def make_five_conv(x, num_filters):
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    return x


# ---------------------------------------------------#
#   Generate Yolo head
# ---------------------------------------------------#
def make_yolo_head(x, num_filters, out_filters):
    y = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    # 255->3, 85->3, 4 + 1 + 80
    y = DarknetConv2D(out_filters, (1, 1))(y)
    return y


# ---------------------------------------------------#
#   Construct FPN network and prediction result
# ---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes):
    inputs = Input(input_shape)
    # ---------------------------------------------------#
    #   retrieve 3 feature maps from backbone network
    #   shape are：
    #   C3 => 52,52,256
    #   C4 => 26,26,512
    #   C5 => 13,13,1024
    # ---------------------------------------------------#
    C3, C4, C5 = darknet_body(inputs)

    # ---------------------------------------------------#
    #   Generate first FPN feature => P5 => (batch_size,13,13,3,85)
    # ---------------------------------------------------#
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    x = make_five_conv(C5, 512)
    P5 = make_yolo_head(x, 512, len(anchors_mask[0]) * (num_classes + 5))

    # 13,13,512 -> 13,13,256 -> 26,26,256
    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(x)

    # 26,26,256 + 26,26,512 -> 26,26,768
    x = Concatenate()([x, C4])
    # ---------------------------------------------------#
    #   Generate second FPN feature => P4 => (batch_size,26,26,3,85)
    # ---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    x = make_five_conv(x, 256)
    P4 = make_yolo_head(x, 256, len(anchors_mask[1]) * (num_classes + 5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x = Concatenate()([x, C3])
    # ---------------------------------------------------#
    #   Generate second FPN feature => P3 => (batch_size,52,52,3,85)
    # ---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    x = make_five_conv(x, 128)
    P3 = make_yolo_head(x, 128, len(anchors_mask[2]) * (num_classes + 5))

    return Model(inputs, [P5, P4, P3])
