# =========================================================================
# Reads Darknet weights and creates Keras model with TF backend.
# =========================================================================

import os
import argparse
import struct
import numpy as np

from tensorflow.keras.utils import plot_model as plot
from model.model_functional import YOLOv3

parser = argparse.ArgumentParser(description='Darknet weights To Keras Converter.')
parser.add_argument('weights_path', help='Path to Darknet weights file.')
parser.add_argument('output_path', help='Path to output Keras model file.')
parser.add_argument(
    '-p',
    '--plot_model',
    default=False,
    help='Plot generated Keras model and save as image.',
    action='store_true')
parser.add_argument(
    '-w',
    '--weights_only',
    default=False,
    help='Save as Keras weights file instead of model file.',
    action='store_true')


# ===================================================================
#   Load weights to yolov3 model
#   Reference:
#   https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/
#   https://github.com/experiencor/keras-yolo3/blob/master/yolo3_one_file_to_detect_them_all.py
#   https://pjreddie.com/media/files/yolov3.weights
#   Input: yolov3 keras model, yolov3.weights
#   Output: yolov3_coco.h5
# ===================================================================
class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)

            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta = self.read_bytes(size)  # bias
                    gamma = self.read_bytes(size)  # scale
                    mean = self.read_bytes(size)  # mean
                    var = self.read_bytes(size)  # variance

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                if len(conv_layer.get_weights()) > 1:
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                print("no convolution #" + str(i))

    def reset(self):
        self.offset = 0


# %%
def _main(args):
    weights_path = os.path.expanduser(args.weights_path)
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)

    output_path = os.path.expanduser(args.output_path)
    assert output_path.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_path)
    output_root = os.path.splitext(output_path)[0]

    # create the model
    model = YOLOv3((None, None, 3), 80)
    print(model.summary())

    # read the model weights
    weight_reader = WeightReader(weights_path)

    # load the weights into the model
    weight_reader.load_weights(model)

    # save model
    if args.weights_only:
        model.save_weights('{}'.format(output_path))
        print('Saved Keras weights to {}'.format(output_path))
    else:
        model.save('{}'.format(output_path))
        print('Saved Keras model to {}'.format(output_path))

    if args.plot_model:
        plot(model, to_file='{}.png'.format(output_root), show_shapes=True)
        print('Saved model plot to {}.png'.format(output_root))


if __name__ == '__main__':
    # run following command (as per current folder structure) on terminal
    # python convert.py data/yolov3.weights data/yolov3_ps.h5
    _main(parser.parse_args())
