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
import time
import logging
import cv2
import mmap
import numpy as np

from thespian.actors import ActorSystem

from object_flow.util.util import Util
from object_flow.ipc.doer import Doer
from object_flow.decoder.display import Display
from object_flow.decoder.video_decoder import VideoDecoder

#==========================================================================================
# VideoManager manages the process of decoding, detection and tracking for one video
# camera.
#==========================================================================================

class VideoManager(Doer):
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def initialize(self, video_name, path, yolo):
        logging.info("+++++++++++++++++++++++++++++++")
        logging.info("VideoManager initializing with video_name %s", video_name)
        self.video_name = video_name
        self.path = path
        self._yolo = yolo

        # open a file for storing modified frames, with detections and tracking
        self.detect_path = "log/detect_" + self.video_name
        self._fd = os.open(self.detect_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC)

        # number of pages 260 should be calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        os.write(self._fd, b'\x00' * mmap.PAGESIZE * 260)
        
        # It seems that there is no way to share memory between processes in
        # Windows, so we use mmap.ACCESS_WRITE that will store the frame on
        # the file. I had hoped that we could share memory.  In Linux, documentation
        # says that memory sharing is possible
        self._detect_buf = mmap.mmap(
            self._fd, 256 * mmap.PAGESIZE, access = mmap.ACCESS_WRITE)
            
    # ----------------------------------------------------------------------------------
    # This method is a callback function when it becomes a listener to the video
    # decoder.  Only after the video decoder is initialize that we have information
    # abuot the width, height and depth of the video being decoded.
    # ----------------------------------------------------------------------------------

    def initialize_mmap(self, mmap_path, width, height, depth):    
        self.mmap_path = mmap_path
        self.width = width
        self.height = height
        self.depth = depth
        self._stop = False
        
        self._total_frames = 0

        # open the mmap file whith the decoded frame. 
        fd = os.open(mmap_path, os.O_RDONLY)
        self._raw_buf = mmap.mmap(fd, 256 * mmap.PAGESIZE, access = mmap.ACCESS_READ)
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def next_frame(self):        
        # call the video decoder to process the next frame
        self.tell(self.video_name, 'next_frame', group = 'decoders')        
        
    # ----------------------------------------------------------------------------------
    # Callback method for the find_bboxes call to the Neural Net
    # ----------------------------------------------------------------------------------

    def detections(self, boxes, confidences, classIDs):
        logging.info(boxes)
        # logging.info(confidences)
        # logging.info(classIDs)

        # if we should draw the input_bbox(es)
        # if (self.cfg.data["video_processor"]["show_input_bbox"]):
        if True:
            for bbox in boxes:
                cv2.rectangle(self._raw_frame, (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]), (0, 250, 0), 2)
                
            # write the frame to the mmap file.  First move the offset to
            # position 0
            self._detect_buf.seek(0)
            size = self._detect_buf.write(self._raw_frame)
            
        self.next_frame()
            
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def process_frame(self, size):
        # logging.info("%s, %s, %s, processing_frame for video %s with size %d",
        #              Util.br_time(), os.getpid(), 'Supervisor', self.video_name, size)

        if self._stop:
            return

        # read the raw frame
        self._raw_buf.seek(0)
        b2 = np.frombuffer(self._raw_buf.read(size), dtype=np.uint8)
        self._raw_frame = b2.reshape((self.height, self.width, self.depth))  # 480, 704, 3
        
        self._total_frames += 1

        if self._total_frames % 20 == 0:
            self.phone(self._yolo, 'find_bboxes', self.video_name, self.mmap_path,
                       self.width, self.height, self.depth, size,
                       callback = 'detections')
        else:
            self.next_frame()
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def start_playback(self):
        display = self.video_name + '_display'
        dp = self.hire(display, Display, self.video_name, group = 'displayers')

        # Starts displaying the video on a new window. For this, add a new listener
        # to the video_decoder and have it callback the initialize method of the
        # Display we have just created above
        self.ask(self.video_name, 'add_listener', self.video_name, dp, 'display',
                 group = 'decoders', callback = 'initialize_mmap', reply_to = dp)
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # TODO: remove video_name as a parameter
    def destroy_window(self, video_name):
        display = self.video_name + '_display'
        self.tell(display, 'destroy_window', group = 'displayers')
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def stop_playback(self):
        self.ask(self.video_name, 'remove_listener', self.video_name,
                 callback = 'destroy_window', group = 'decoders')
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def run(self):
        # hire a new video decoder named 'self.video_name'
        vd = self.hire(self.video_name, VideoDecoder, self.video_name, self.path,
                       group = 'decoders')
        
        # add this manager as a listener to the decoded video frames
        self.phone(vd, 'add_listener', self.video_name + '_manager', self.myAddress,
                   'process_frame', callback = 'initialize_mmap')
        self.post(vd, 'start_processing')

