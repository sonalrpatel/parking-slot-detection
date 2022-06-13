# =========================================================================
# Implementation of YOLOv3 architecture
# =========================================================================
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, Model
from configs import YOLO_LAYER_WITH_NAMES

"""
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""


def DarknetConv2D_BN_Leaky(inputs, n_filters, kernel_size=(3, 3), down_sample=False, bn_act=True, layer_idx=None):
    strides = (2, 2) if down_sample else (1, 1)
    padding = 'valid' if down_sample else 'same'
    use_bias = False if bn_act else True

    x = Conv2D(n_filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
               kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=l2(5e-4),
               name='conv_' + str(layer_idx) if YOLO_LAYER_WITH_NAMES else None)(inputs)
    if bn_act:
        x = BatchNormalization(name='bnorm_' + str(layer_idx) if YOLO_LAYER_WITH_NAMES else None)(x)
        x = LeakyReLU(alpha=0.1, name='leaky_' + str(layer_idx) if YOLO_LAYER_WITH_NAMES else None)(x)

    return x


def ResidualBlock(inputs, n_filters, n_repeats=1, layer_idx=None):
    x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x = DarknetConv2D_BN_Leaky(x, n_filters, down_sample=True, layer_idx=layer_idx)
    for i in range(n_repeats):
        y = DarknetConv2D_BN_Leaky(x, n_filters // 2, kernel_size=(1, 1), layer_idx=layer_idx + 1 + (i * 3))
        y = DarknetConv2D_BN_Leaky(y, n_filters, layer_idx=layer_idx + 2 + (i * 3))
        x = Add()([x, y])  # layer_idx=layer_idx + 3 + (i*3)

    return x


def DarkNet53(inputs, layer_idx=0):
    # Layer 0
    x = DarknetConv2D_BN_Leaky(inputs, 32, layer_idx=layer_idx)
    # Layer 1 (+3*1) => 4
    x = ResidualBlock(x, 64, n_repeats=1, layer_idx=layer_idx + 1)
    # Layer 5 (+3*2) => 11
    x = ResidualBlock(x, 128, n_repeats=2, layer_idx=layer_idx + 5)
    # Layer 12 (+3*8) => 36
    x = ResidualBlock(x, 256, n_repeats=8, layer_idx=layer_idx + 12)
    skip_36 = x
    # Layer 37 (+3*8) => 61
    x = ResidualBlock(x, 512, n_repeats=8, layer_idx=layer_idx + 37)
    skip_61 = x
    # Layer 62 (+3*4) => 74
    x = ResidualBlock(x, 1024, n_repeats=4, layer_idx=layer_idx + 62)

    return skip_36, skip_61, x


def UpSampleConv(inputs, n_filters, layer_idx=0):
    x = inputs
    idx = 0
    if not tf.is_tensor(inputs):
        x = DarknetConv2D_BN_Leaky(inputs[0], n_filters, kernel_size=(1, 1), layer_idx=layer_idx)
        x = UpSampling2D(2)(x)                      # layer_idx=layer_idx + 1
        x = Concatenate(axis=-1)([x, inputs[1]])    # layer_idx=layer_idx + 2
        idx = 3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1), layer_idx=layer_idx + idx)         # 512, 1x1
    x = DarknetConv2D_BN_Leaky(x, n_filters * 2, layer_idx=layer_idx + idx + 1)                     # 1024, 3x3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1), layer_idx=layer_idx + idx + 2)     # 512, 1x1
    x = DarknetConv2D_BN_Leaky(x, n_filters * 2, layer_idx=layer_idx + idx + 3)                     # 1024, 3x3
    x = DarknetConv2D_BN_Leaky(x, n_filters, kernel_size=(1, 1), layer_idx=layer_idx + idx + 4)     # 512, 1x1

    return x


def ScalePrediction(inputs, n_filters, num_classes, layer_idx=0):
    x = DarknetConv2D_BN_Leaky(inputs, n_filters, layer_idx=layer_idx)  # 13x13x1024/26x26x512/52x52x256, 3x3
    x = DarknetConv2D_BN_Leaky(x, 3 * (num_classes + 5), kernel_size=(1, 1), bn_act=False, layer_idx=layer_idx + 1)
                                                                        # 13x13x255/26x26x255/52x52x255, 1x1

    return x


def YOLOv3(input_shape=(416, 416, 3), num_classes=80):
    x = Input(input_shape)

    def model(inputs, num_classes):
        # Layer 0 => 74
        skip_36, skip_61, x_74 = DarkNet53(inputs, layer_idx=0)

        # Layer 75 => 79
        x_79 = UpSampleConv(x_74, 512, layer_idx=75)
        # Layer 80 => 81
        y_lbbox_81 = ScalePrediction(x_79, 1024, num_classes, layer_idx=80)

        # Layer 84 => 91
        x_91 = UpSampleConv([x_79, skip_61], 256, layer_idx=84)
        # Layer 92 => 93
        y_mbbox_93 = ScalePrediction(x_91, 512, num_classes, layer_idx=92)

        # Layer 96 => 103
        x_103 = UpSampleConv([x_91, skip_36], 128, layer_idx=96)
        # Layer 104 => 105
        y_sbbox_105 = ScalePrediction(x_103, 256, num_classes, layer_idx=104)

        return [y_lbbox_81, y_mbbox_93, y_sbbox_105]

    return Model(inputs=[x], outputs=model(x, num_classes))


def _main():
    num_classes = 80
    image_size = 416
    image_shape = (image_size, image_size, 3)

    # define the model
    model = YOLOv3(image_shape, num_classes)

    print(model.summary())


if __name__ == "__main__":
    _main()
