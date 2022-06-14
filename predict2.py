# =========================================================================
#   predict2.py integrates functions such as single image prediction, video/camera detection, FPS test and
#       directory traversal detection.
#   It is integrated into a py file, and the mode is modified by specifying the mode.
# =========================================================================

import os
import cv2
import time
import colorsys
import argparse
from tqdm import tqdm
from PIL import ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from access_dict_by_dot import AccessDictByDot

from model.model_functional import YOLOv3
from utils.utils import *
from utils.utils_bbox import *


parser = argparse.ArgumentParser(description='Objects detection on an Image using Yolov3')
parser.add_argument(
    '-w',
    '--weight_path',
    default=None,
    help='Path to the weights file.')
parser.add_argument(
    '-c',
    '--classes_path',
    default=None,
    help='Path to the classes file.')
parser.add_argument(
    '-i',
    '--image_path',
    default=None,
    help='Path to the image file.')


class YoloDecode(object):
    # =====================================================================
    #   Initialize yolo result
    # =====================================================================
    def __init__(self, args):
        # =====================================================================
        #   To use your own trained model for prediction, you must modify weight_path and classes_path!
        #   weight_path points to the weights file under the logs folder,
        #       classes_path points to the txt under data folder.
        #
        #   After training, there are multiple weight files in the logs folder,
        #       and you can select the validation set with lower loss.
        #   The lower loss of the validation set does not mean that the mAP is higher,
        #       it only means that the weight has better generalization performance on the validation set.
        #   If the shape does not match, pay attention to the modification of the model_path
        #       and classes_path parameters during training
        # =====================================================================
        self.weight_path = args.weight_path if args.weight_path is not None else 'data/yolov3_coco.h5'
        self.classes_path = args.classes_path if args.classes_path is not None else 'data/coco_classes.txt'

        # =====================================================================
        #   anchors_path represents the txt file corresponding to the a priori box, which is generally not modified.
        #   anchors_mask is used to help the code find the corresponding a priori box and is generally not modified.
        # =====================================================================
        self.anchors_path = 'data/yolo_anchors.txt'
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        # =====================================================================
        #   The size of the input image, which must be a multiple of 32.
        # =====================================================================
        self.input_shape = [416, 416]
        self.input_image_shape = Input([2, ], batch_size=1)

        # =====================================================================
        #   Only prediction boxes with scores greater than confidence will be kept
        # =====================================================================
        self.conf_thresh = 0.2

        # =====================================================================
        #   nms_iou size used for non-maximum suppression
        # =====================================================================
        self.nms_iou_thresh = 0.5
        self.max_boxes = 100

        # =====================================================================
        #   This variable is used to control whether to use letterbox_image
        #       to resize the input image without distortion,
        #   After many tests, it is found that the direct resize effect of closing letterbox_image is better
        # =====================================================================
        self.letterbox_image = True

        # =====================================================================
        #   Get the number of kinds and a priori boxes
        # =====================================================================
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)

        # =====================================================================
        #   Picture frame set different colors
        # =====================================================================
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # =====================================================================
        #   Create a yolo model
        # =====================================================================
        self.model_body = YOLOv3((None, None, 3), self.num_classes)

        # =====================================================================
        #   Load model weights
        # =====================================================================
        weight_path = os.path.join(os.path.dirname(__file__), self.weight_path)
        assert weight_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        assert os.path.exists(weight_path), 'Keras model or weights file does not exist.'

        self.model_body.load_weights(weight_path, by_name=True, skip_mismatch=True)
        print('{} model, anchors, and classes loaded.'.format(weight_path))

        # =====================================================================
        #   In the DecodeBox function, we will post-process the prediction results
        #   The content of post-processing includes decoding, non-maximum suppression, threshold filtering, etc.
        # =====================================================================
        outputs = Lambda(
            DecodeBox,
            output_shape=(1,),
            name='yolo_eval',
            arguments={
                'anchors': self.anchors,
                'num_classes': self.num_classes,
                'input_shape': self.input_shape,
                'anchor_mask': self.anchors_mask,
                'conf_thresh': self.conf_thresh,
                'nms_iou_thresh': self.nms_iou_thresh,
                'max_boxes': self.max_boxes,
                'letterbox_image': self.letterbox_image
            }
        )([*self.model_body.output, self.input_image_shape])

        # =====================================================================
        #   Construct model with DecodeBox layer
        # =====================================================================
        #      image                              input_image_shape
        #    (x,y :RGB)   -------------------->     (2, :Tensor)
        #        |                                       |
        #        V                                       |
        #    image_data                                  |
        #   (1,416,416,3)                                V
        #        |          yolo pred (raw)           DecodeBox
        #        V           (1,13,13,255)     (decoding raw yolo prediction,
        #    model_body  --> (1,26,26,255) -->   correction of bboxes size,   --> draw image
        #                    (1,52,52,255)        non maximal suppression)        with bboxes
        # =====================================================================
        self.model_decode = Model([self.model_body.input, self.input_image_shape], outputs)

    # =====================================================================
    #   Preprocess image
    # =====================================================================
    def __preprocess_image(self, image):
        # =====================================================================
        #   Convert the image to an RGB image here to prevent an error in the prediction of the grayscale image.
        #   The code only supports prediction of RGB images, all other types of images will be converted to RGB
        # =====================================================================
        image = convert2rgb(image)

        # =====================================================================
        #   Add gray bars to the image to achieve undistorted resize
        #   You can also directly resize for identification
        # =====================================================================
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        # =====================================================================
        #   Normalize the image
        # =====================================================================
        image_data = preprocess_input(np.array(image_data, dtype='float32'))
        return image_data

    # =====================================================================
    #   Draw boxes on input image
    # =====================================================================
    def __draw_boxes(self, image, out_boxes, out_scores, out_classes):
        # =====================================================================
        #   Set font and border thickness
        # =====================================================================
        font = ImageFont.truetype(font='font/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        # =====================================================================
        #   Image drawing
        # =====================================================================
        for e, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box = out_boxes[e]
            score = out_scores[e]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

    # =====================================================================
    #   Detect pictures
    # =====================================================================
    def detect_image(self, image, mode=None):
        # =====================================================================
        #   Preprocess image and Add the batch_size dimension
        # =====================================================================
        image_data = self.__preprocess_image(image)

        image_data = np.expand_dims(image_data, 0)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)

        # =====================================================================
        #   Feed the image into the network to make predictions!
        # =====================================================================
        out_boxes, out_scores, out_classes = self.model_decode([image_data, input_image_shape])
        if mode != "dir_predict":
            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # =====================================================================
        #   Draw bounding boxes on the image using labels
        # =====================================================================
        self.__draw_boxes(image, out_boxes, out_scores, out_classes)
        return image

    # =====================================================================
    #   Calculate number of image files processed per second
    # =====================================================================
    def get_FPS(self, image, test_interval):
        # =====================================================================
        #   Preprocess image and Add the batch_size dimension
        # =====================================================================
        image_data = self.__preprocess_image(image)
        image_data = np.expand_dims(image_data, 0)

        # =====================================================================
        #   Feed the image into the network to make predictions!
        # =====================================================================
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.model_decode([image_data, input_image_shape])

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.model_decode([image_data, input_image_shape])

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    # =====================================================================
    #   Detect pictures
    # =====================================================================
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        # =====================================================================
        #   Preprocess image and Add the batch_size dimension
        # =====================================================================
        image_data = self.__preprocess_image(image)
        image_data = np.expand_dims(image_data, 0)

        # =====================================================================
        #   Feed the image into the network to make predictions!
        # =====================================================================
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.model_decode([image_data, input_image_shape])

        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            try:
                score = str(out_scores[i].numpy())
            except:
                score = str(out_scores[i])
            top, left, bottom, right = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))

        f.close()
        return


def _main(args):
    # =====================================================================
    #   Create an object of yolo result class
    # =====================================================================
    yolo = YoloDecode(args)

    # =====================================================================
    #   mode is used to specify the mode of the test:
    #   'predict' means single image prediction. If you want to modify the prediction process, such as saving images,
    #       intercepting objects, etc., you can read the detailed notes below first.
    #   'video' means video detection, you can call the camera or video for detection, see the notes below for details.
    #   'fps' means test fps, the image used is street.jpg in img, see the notes below for details.
    #   'dir_predict' means to traverse the folder to detect and save. By default, the img folder is traversed and
    #       the img_out folder is saved. For details, see the notes below.
    # =====================================================================
    mode = "predict"

    # =====================================================================
    #   video_origin_path is used to specify the path of the video, when video_origin_path = 0, it means to detect
    #       the camera.
    #   If you want to detect the video, set it as video_origin_path = "xxx.mp4", which means to read the xxx.mp4 file
    #       in the root directory.
    #   video_save_path indicates the path where the video is saved, when video_save_path = "" it means not to save.
    #   If you want to save the video, set it as video_save_path = "yyy.mp4", which means that it will be saved as
    #       a yyy.mp4 file in the root directory.
    #   video_fps is the fps of the saved video.
    #   video_origin_path, video_save_path and video_fps are only valid when mode = 'video'
    #   When saving the video, you need ctrl+c to exit or run to the last frame to complete the complete save step.
    # =====================================================================
    video_origin_path = 0
    video_save_path = ""
    video_fps = 25.0

    # =====================================================================
    #   test_interval is used to specify the number of image detections when measuring fps
    #   In theory, the larger the test_interval, the more accurate the fps.
    # =====================================================================
    test_interval = 100

    # =====================================================================
    #   dir_origin_path specifies the folder path of the image used for detection
    #   dir_save_path specifies the save path of the detected image
    #   dir_origin_path and dir_save_path are only valid when mode = 'dir_predict'
    # =====================================================================
    dir_origin_path = "data/demo/train/"
    dir_save_path = "data_out/"

    # =====================================================================
    #   If you want to save the detected image, use image_out.save("img.jpg") to save it, and modify it directly in
    #       predict2.py.
    #   If you want to get the coordinates of the prediction frame, you can enter the yolo.detect_image function and
    #       read the four values of top, left, bottom, and right in the drawing part.
    #   If you want to use the prediction frame to intercept the target, you can enter the yolo.detect_image function,
    #       and use the obtained four values of top, left, bottom, and right in the drawing part.
    #   Use the matrix method to intercept the original image.
    #   If you want to write extra words on the prediction map, such as the number of specific targets detected,
    #       you can enter the yolo.detect_image function and judge the predicted_class in the drawing part,
    #       For example, judging if predicted_class == 'car': can judge whether the current target is a car,
    #       and then record the number. Use draw.text to write.
    # =====================================================================
    if mode == "predict":
        import os
        image_path = args.image_path if args.image_path is not None else "data/sample/apple.jpg"
        image = Image.open(os.path.join(os.path.dirname(__file__), image_path))
        image_out = yolo.detect_image(image)
        image_out.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_origin_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("The camera (video) cannot be read correctly, please pay attention to whether the camera"
                             "is installed correctly (whether the video path is correctly filled in).")

        fps = 0.0
        while True:
            t1 = time.time()
            # read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # format conversion, BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # test
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('data/fruits.webp')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                image_out = yolo.detect_image(image, mode)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                image_out.save(os.path.join(dir_save_path, img_name))

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")


if __name__ == '__main__':
    # run following command (as per current folder structure) on terminal
    # python predict2.py [-i] <image_path>
    # python predict2.py -w data/trained_weights_final.h5 -c data/demo/train/_classes.txt -i data/demo/train/20160725-3-1.jpg
    # python predict2.py -w data/ep064-loss0.431-val_loss0.408.h5 -c data/demo/train/_classes.txt
    dictionary = {
        'weight_path' : "data/ep064-loss0.431-val_loss0.408.h5",
        'classes_path' : "data/demo/train/_classes.txt",
        'image_path' : "data/demo/train/20160725-3-1.jpg"
    }
    args = AccessDictByDot.load(dictionary)
    _main(args)
    # _main(parser.parse_args())
