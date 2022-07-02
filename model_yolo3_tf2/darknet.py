from functools import wraps

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Conv2D, LeakyReLU, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from utils.utils import compose


# ------------------------------------------------------#
#   Single DarknetConv2D Block
#   DarknetConv2D
# ------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer': RandomNormal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


# ---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + LeakyReLU
# ---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


# ---------------------------------------------------------------------#
#   Residual Block
#   Use ZeroPadding2D and a Conv with strides of 2 to reduce image height and width
#   Loop num_blocks to produce multiple residual blocks
# ---------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(y)
        x = Add()([x, y])
    return x


# ---------------------------------------------------#
#   darknet53 backbone network
#   inputs image size = 416x416x3
#   outputs 3 feature maps
# ---------------------------------------------------#
def darknet_body(x):
    # 416,416,3 -> 416,416,32
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    # 416,416,32 -> 208,208,64
    x = resblock_body(x, 64, 1)
    # 208,208,64 -> 104,104,128
    x = resblock_body(x, 128, 2)
    # 104,104,128 -> 52,52,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 52,52,256 -> 26,26,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 26,26,512 -> 13,13,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3
