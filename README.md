# parking-slot-detection

[Introduction](#Introduction)

[References](#References)

- [Paper](#Paper)
- [Github](#Github)
- [Blog](#Blog)
- [Dataset](#Dataset)

[Folder Structure](#Folder-Structure)

[Sample Results](#Sample-Results)

[TODOs](#TODOs)

## Introduction

This repository is an implementation of parking slot detection in AVM (around view images) using deep learning.
The implementation is based on following [References](#References).

## References

### Paper
around view image

- [Research Review on Parking Space Detection Method](https://www.mdpi.com/2073-8994/13/1/128/pdf)
- [Vacant Parking Slot Detection in the Around View Image Based on Deep Learning](https://www.mdpi.com/1424-8220/20/7/2138/htm)
- [Parking Slot Detection on Around-View Images Using DCNN](https://www.frontiersin.org/articles/10.3389/fnbot.2020.00046/full)
- [Vision-Based Parking-Slot Detection: A DCNN-Based Approach and a Large-Scale Benchmark Dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8412601)
- [Real Time Detection Algorithm of Parking Slot Based on Deep Learning and Fisheye Image](https://iopscience.iop.org/article/10.1088/1742-6596/1518/1/012037/pdf)
- [A Deep-Learning Approach for Parking Slot Detection on Surround-View Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8813777)
- [Context-Based Parking Slot Detection With a Realistic Dataset](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9199853)
- [End to End Trainable One Stage Parking Slot Detection Integrating Global and Local Information](https://arxiv.org/ftp/arxiv/papers/2003/2003.02445.pdf)
- [PSDet: Efficient and Universal Parking Slot Detection](https://arxiv.org/pdf/2005.05528.pdf)

parking lot image

- [Automated Vehicle Parking Slot Detection System Using Deep Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9076491&tag=1)
- [Automated Parking Space Detection Using Convolutional Neural Networks](https://arxiv.org/pdf/2106.07228.pdf)
- [An Elaborative Study of Smart Parking Systems](https://www.ijert.org/research/an-elaborative-study-of-smart-parking-systems-IJERTV10IS100056.pdf)
- [Autonomous Parking-Lots Detection with Multi-Sensor Data Fusion](https://www.techscience.com/cmc/v66n2/40666)
- [Automatic Parking System Based on Improved Neural Network Algorithm and Intelligent Image Analysis](https://downloads.hindawi.com/journals/cin/2021/4391864.pdf)

### Github
around view image

- [awesome-parking-slot-detection](https://github.com/lymhust/awesome-parking-slot-detection)
- [VPS-Net](https://github.com/weili1457355863/VPS-Net)
- [context-based-parking-slot-detect](https://github.com/dohoseok/context-based-parking-slot-detect)
- [Parking-slot-detection](https://github.com/wuzzh/Parking-slot-detection)
- [Parking-slot-dataset](https://github.com/wuzzh/Parking-slot-dataset)
- [MarkToolForParkingLotPoint](https://github.com/Teoge/MarkToolForParkingLotPoint)

object detection using yolo

- [object-detection-yolov3](https://github.com/sonalrpatel/object-detection-yolo)

### Blog
around view image

- [Vision-based Parking-slot Detection: A DCNN-based Approach and A Large-scale Benchmark Dataset](https://cslinzhang.github.io/deepps/)

parking lot image

- [Searching for a Parking Spot? AI Got It](https://blogs.nvidia.com/blog/2019/09/11/drive-labs-ai-parking/)
- [AI Can Detect Open Parking Spaces](https://news.developer.nvidia.com/ai-algorithm-aims-to-help-you-find-a-parking-spot/)
- [Parking Lot Vehicle Detection Using Deep Learning](https://medium.com/geoai/parking-lot-vehicle-detection-using-deep-learning-49597917bc4a)
- [Parking Space Detection Using Deep Learning](https://medium.com/the-research-nest/parking-space-detection-using-deep-learning-9fc99a63875e)
- [Parking Occupancy Detection Using AI and ML](https://visionify.ai/parking-occupancy-detection-using-ai-ml/)

### Dataset
around view image

- [ps2.0] [Tongji Parking-slot Dataset 2.0: A Large-scale Benchmark Dataset](https://cslinzhang.github.io/deepps/)
- [Context-Based Parking Slot Detect Dataset](https://github.com/dohoseok/context-based-parking-slot-detect)

## Folder Structure

Explanation about folders and files.

- __data__ - contains the datasets, annotation files, and class details
  - {dataset folder name}
    - train
    - val
    - test
    - train_annotation.txt
    - val_annotation.txt
    - test_annotation.txt
  - ps_classes.txt
- __dataloader__
  - dataloader.py - custom data generator
- __loss__
  - loss_functional.py - loss is written in a function
  - loss_subclass.py - loss is written under a class
- __model__
  - darknet.py - backbone
  - model_functional.py - functional model
  - model_subclass.py - model sub-classing
- __model_yolo3_tf2__ - yolov3 model reference from [yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
- __model_data__ - contains the .cfg, .weights, .h5, font files
- __model_img__ - contains the architecture images
- __notebook__ - contains the jupyter / google colab notebook file
- __utils__
  - callbacks.py
  - dataloader.py - reference from [yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
  - utils.py
  - utils_bbox.py
  - utils_metric.py
- __configs.py__
- __convert.py__
- __predict.py__
- __train.py__
- __convert0.py__ - reference from [yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
- __predict0.py__ - reference from [yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
- __train0.py__ - reference from [yolov3-tf2](https://github.com/zzh8829/yolov3-tf2)
- __psmat2txt.py__ - generates annotations in __YOLO v3 Keras TXT__ format from __mat__ files
- __visbbox.py__ - visualise bounding boxes

## Sample Results

Sample results of parking slot head detection with [VPS-Net](https://github.com/weili1457355863/VPS-Net) as reference

![Parking slot head](outputs/demo/20160725-3-97.jpg)
![Parking slot head](outputs/demo/20160725-3-647.jpg)

## TODOs

- find complete parking slot
- improve performance by referring other papers
