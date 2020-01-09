# -*- coding: utf-8 -*-
# encoding: utf-8
# encoding: iso-8859-1
# encoding: win-1252

##########################################################################################
# @author Rodrigo Botafogo
#
# Copyright (C) 2019 Rodrigo Botafogo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019
##########################################################################################

# needed to open the mmap file
import os
import mmap
import math

import logging
import time

import tensorflow as tf
import numpy as np
import cv2

from object_flow.util.util import Util
from object_flow.ipc.doer import Doer

from object_flow.nn.yolov3_tf2.models import YoloV3
from object_flow.nn.yolov3_tf2.dataset import transform_images
# from object_flow.nn.yolov3_tf2.models import YoloV3Tiny

class YoloTf2(Doer):
    
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        
        logging.info("yolotf2 initialization started")
        
        # base path to the yolo directory with weights, names and config
        yolo_dir = 'object_flow/nn/resources/'
        coco_dir = os.path.sep.join([yolo_dir, 'yolo-coco'])
        names_path = os.path.sep.join([coco_dir, 'coco.names'])
        
        self.LABELS = open(names_path).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                        dtype="uint8")

        logging.info("set memory growth to True")
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # TODO: number of classes should be on the nn_cfg file
        self.yolo = YoloV3(classes=80)
        self.yolo.load_weights('object_flow/nn/resources/checkpoints/yolov3.tf')
        logging.info("weights loaded")
            
        class_names = [c.strip() for c in open(names_path).readlines()]
        logging.info("classes loaded")

        # file descriptors opened where frames are stored
        self._fd = {}
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def __initialize__(self, confidence, threshold):
        logging.info("Yolo, setting confidence to %f", confidence)
        logging.info("Yolo, setting threshold to %f", threshold)
        
        self.min_confidence = confidence
        self.threshold = threshold
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    # def find_bboxes(self, frame, width, height):
    def find_bboxes(self, name, file_name, width, height, depth, size):

        # open the file descriptor if not already opened
        if not name in self._fd:
            self._fd[name] = os.open(file_name, os.O_RDONLY)

        # read the image
        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        npage = math.ceil((width * height * depth)/ 4000) + 10
        
        self._buf = mmap.mmap(
            self._fd[name], mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)
        self._buf.seek(0)
        b2 = np.frombuffer(self._buf.read(size), dtype=np.uint8)
        frame = b2.reshape((height, width, depth))

        # Resize the frame to 416 x 416 as required by Yolo and then convert it
        # to RGB
        frame = cv2.resize(frame, (416, 416))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
    
        img = frame

        # resize the image to 416 x 416 (seems to be the dimention required by yolo)
        # imge = np.array(img).reshape(-1, 416, 416, 3)
        imge = tf.expand_dims(img, 0)
        imge = transform_images(imge, 416)

        start_time = time.time()
        bboxes, scores, classes, nums = self.yolo.predict(imge)
        bboxes, objectness, classes, nums = bboxes[0], scores[0], classes[0], nums[0]
        wh = np.flip(img.shape[0:2])

        # Constants needed to resize the identified bboxes to the original frame size
        kw = width/416
        kh = height/416
        
        for i in range(nums):
            # taking only class 0 person... needs to be configurable
            if classes[i] == 0:
                if objectness[i] > self.min_confidence:
                    x1y1 = tuple((np.array(bboxes[i][0:2]) * wh).astype(np.int32))
                    x2y2 = tuple((np.array(bboxes[i][2:4]) * wh).astype(np.int32))
                    
                    box = [x1y1[0], x1y1[1], x2y2[0], x2y2[1]]
                    box = box * np.array([kw, kh, kw, kh])
                    boxes.append(box.astype(np.int32))
                    
                    confidences.append(objectness[i])
                    classIDs.append(classes[i])

        return boxes, confidences, classIDs
