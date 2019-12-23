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

from object_flow.ipc.doer import Doer

from object_flow.util.display import Display

from object_flow.decoder.video_decoder import VideoDecoder
from object_flow.flow.item import Item
from object_flow.flow.setting import Setting

#==========================================================================================
# FlowManager manages the process of decoding, detection and tracking for one video
# camera.  Every FlowManager has a Setting, i.e., the camera view with all the added
# information, such as the counting lines, the identified items, the bounding boxes,
# etc.
#==========================================================================================

class FlowManager(Doer):
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()

        # total number of frames processed by this FlowManager
        self._total_frames = 0
        
        # list of listeners interested to get a message everytime a new frame is
        # loaded
        self._listeners = {}

        self.playback = False
        self.playback_started = False
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def initialize(self, video_name, path, yolo):
        self.video_name = video_name
        self.path = path
        self._yolo = yolo
        self._setting = Setting()

        logging.info("initializing flow_manager %s in path %s", video_name, path)
        
        self.run()

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
        
        # open the mmap file whith the decoded frame. 
        fd = os.open(mmap_path, os.O_RDONLY)
        self._raw_buf = mmap.mmap(fd, 256 * mmap.PAGESIZE, access = mmap.ACCESS_READ)

        logging.info("mmap file for %s opened", self.video_name)

        # now that the mmap file has been initialized, we can call 'start_processing'
        self.post(self.vd, 'start_processing')
        self.post(self.parent_address, 'flow_manager_initialized', self.video_name)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def hired(self, hiree_name, hiree_group, hiree_address):
        if hiree_group == 'decoders':
            logging.info("decoder for %s created", self.video_name)
            self.phone(hiree_address, 'add_listener', self.video_name + '_manager',
                       self.myAddress, 'process_frame', callback = 'initialize_mmap')
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def run(self):
        # hire a new video decoder named 'self.video_name'
        self.vd = self.hire(self.video_name, VideoDecoder, self.video_name, self.path,
                            group = 'decoders')
        
    # ----------------------------------------------------------------------------------
    # Adds a new listener to this flow_manager. When a new listener is added it can
    # use the values of width, height and depth already initialized from the camera
    # ----------------------------------------------------------------------------------

    def add_listener(self, name, address, callback):
        logging.info("adding listener to flow_manager %s with name %s", self.video_name,
                     name)
        self._listeners[name] = (address, callback)
        return (self.mmap_path, self.width, self.height, self.depth)
    
    # ----------------------------------------------------------------------------------
    # Removes the listener
    # ----------------------------------------------------------------------------------

    def remove_listener(self, name):
        logging.info("listener %s removed from flow_manager %s", name, self.video_name)
        del self._listeners[name]
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def next_frame(self):

        # notify every listener that we have a new frame and give it the
        # buffer size
        for name, listener in self._listeners.items():
            # listener[0]: doer's address
            # listener[1]: doer's method to call
            self.post(listener[0], 'base_image', self._buf_size)
            self.post(listener[0], 'overlay_bboxes', self._setting.items)
            self.post(listener[0], listener[1], self._buf_size)
        
        # call the video decoder to process the next frame
        self.tell(self.video_name, 'next_frame', group = 'decoders')
        
    # ----------------------------------------------------------------------------------
    # Callback method for the find_bboxes call to the Neural Net
    # ----------------------------------------------------------------------------------

    def detections(self, boxes, confidences, classIDs):
        # if we should draw the input_bbox(es)
        # if (self.cfg.data["video_processor"]["show_input_bbox"]):
        if True:
            self._setting.add_detections(boxes, confidences, classIDs)
            
        self.next_frame()
            
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def process_frame(self, size):
        # read the raw frame
        self._buf_size = size
        self._raw_buf.seek(0)
        b2 = np.frombuffer(self._raw_buf.read(size), dtype=np.uint8)
        self._raw_frame = b2.reshape((self.height, self.width, self.depth))  # 480, 704, 3
        
        self._total_frames += 1

        if self._total_frames % 10 == 0:
            self.phone(self._yolo, 'find_bboxes', self.video_name, self.mmap_path,
                       self.width, self.height, self.depth, size,
                       callback = 'detections')
        else:
            self.next_frame()
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _add_listener(self, v2):
        # Starts displaying the video on a new window. For this, add a new listener
        # to the video_decoder and have it callback the initialize method of the
        # Display we have just created above
        self.add_listener(self.video_name, self._dp, 'display')
        self.playback_started = True

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # PLAYBACK MANAGEMENT
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def start_playback(self):

        display  = self.video_name + '_display'
        self._dp = self.hire(display, Display, self.video_name, group = 'displayers')

        logging.info("starting playback for video %s", self.video_name)
        
        # initialize the display
        self.phone(self._dp, 'initialize_mmap', self.mmap_path, self.width,
                   self.height, self.depth, callback = '_add_listener')
                
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
        
        if self.playback == False:
            return
        elif self.playback_started == False:
            self.post(self.myAddress, 'stop_playback')
            return

        logging.info("stopping playback for video %s", self.video_name)
        
        # self.ask(self.video_name, 'remove_listener', self.video_name,
        #          callback = 'destroy_window', group = 'decoders')
        self.remove_listener(self.video_name)
        self.destroy_window(self.video_name)
        
        self.playback = False
        self.playback_started = False
        
