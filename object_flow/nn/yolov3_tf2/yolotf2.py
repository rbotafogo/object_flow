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

import os
import logging
import time

import tensorflow as tf
import numpy as np
import cv2

from object_flow.ipc.doer import Doer
from object_flow.nn.yolov3_tf2.models import YoloV3
from object_flow.nn.yolov3_tf2.dataset import transform_images
from object_flow.util.mmap_frames import MmapFrames
from object_flow.util.mmap_bboxes import MmapBboxes

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
        logging.info("loading weights from file 'object_flow/nn/resources/checkpoints/yolov3.tf'")
        self.yolo.load_weights('object_flow/nn/resources/checkpoints/yolov3.tf')
        logging.info("weights loaded")
            
        class_names = [c.strip() for c in open(names_path).readlines()]
        logging.info("classes loaded")

        # Yolo performs detection for different videos, so it has a list of all
        # videos and for each video.
        self.videos = {}
        
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
        
        # mmap file for writing detected object's bounding boxes
        self._mmap_bbox = MmapBboxes()
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def terminate(self):
        super().terminate()
        self._mmap.close()
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # SERVICES
    
    # ----------------------------------------------------------------------------------
    # Registers a video with this tracker
    # ----------------------------------------------------------------------------------

    def register_video(self, video_name, video_id, width, height, depth):
        self.videos[video_name] = {}
        self.videos[video_name]['video_id'] = video_id
        self.videos[video_name]['width'] = width
        self.videos[video_name]['height'] = height
        self.videos[video_name]['depth'] = depth
        self.videos[video_name]['frame_size'] = width * height * depth
        # mmap file for accessing video frames
        self.videos[video_name]['frames'] = MmapFrames(video_name, width, height, depth)
        self.videos[video_name]['frames'].open_read()
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def find_bboxes(self, video_name, frame_index):

        frame_number, frame = self.videos[video_name]['frames'].read_data(frame_index)
        video_id = self.videos[video_name]['video_id']
        width = self.videos[video_name]['width']
        height = self.videos[video_name]['height']
        
        self._mmap_bbox.open_write(video_name, video_id)
        
        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
    
        # resize the image to 416 x 416 (seems to be the dimention required by yolo)
        # to RGB
        frame = cv2.resize(frame, (416, 416))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imge = transform_images(tf.expand_dims(frame, 0), 416)

        bboxes, scores, classes, nums = self.yolo.predict(imge)
        bboxes, objectness, classes, nums = bboxes[0], scores[0], classes[0], nums[0]

        wh = np.flip(frame.shape[0:2])

        # Constants needed to resize the identified bboxes to the original frame size
        kw = width/416
        kh = height/416
        
        self._mmap_bbox.set_pointer(video_id)
        
        for i in range(nums):
            # taking only class 0 person... needs to be configurable
            if classes[i] == 0:
                if objectness[i] > self.min_confidence:
                    x1y1 = tuple((np.array(bboxes[i][0:2]) * wh).astype(np.int32))
                    x2y2 = tuple((np.array(bboxes[i][2:4]) * wh).astype(np.int32))
                    
                    box = [x1y1[0], x1y1[1], x2y2[0], x2y2[1]]
                    box = box * np.array([kw, kh, kw, kh])

                    self._mmap_bbox.write_bbox(video_name,
                        box.astype(np.int32), objectness[i], classes[i])
                    
                    boxes.append(box.astype(np.int32))
                    
                    confidences.append(objectness[i])
                    classIDs.append(classes[i])

        self._mmap_bbox.close()
        
        return boxes, confidences, classIDs
