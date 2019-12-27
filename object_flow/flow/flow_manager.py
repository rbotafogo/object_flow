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
import math

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

    def initialize(self, cfg, yolo):
        self.cfg = cfg
        self.video_name = cfg.analyser_id
        self.path = cfg.data['io']['input']
        self._yolo = yolo
        self._setting = Setting()

        logging.info("initializing flow_manager %s in path %s", self.video_name,
                     self.path)
        
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
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        npage = math.ceil((self.width * self.height * self.depth)/ 4000) + 10
        fd = os.open(mmap_path, os.O_RDONLY)
        self._raw_buf = mmap.mmap(fd, mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)

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
    # This method notifies all listeners that we have a new frame processed. It sends
    # the following messages to the listeners:
    # 'base_image': with the size of the buffer (mmap) where the image is
    # 'overlay_bboxes': with all the detection boxes found
    # registered method.... TODO: should change this!!!
    # Finally, this method sends to the decoder the 'next_frame' message for it to
    # decode a new frame.  This closes the processing loop: 1) decoder decodes a frame;
    # 2) decoder calls 'process_frame' from flow_manager; 3) flow_manager does whatever
    # it needs to to with the frame; 4) flow_manager calls 'next_frame' (this method);
    # 5) 'next_frame' calls back onto the decoder (step 1 above)
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
    # Callback method for the 'find_bboxes' call to the Neural Net.  This callback is
    # registered by method 'process_frame'. This method simply adds the detected
    # items into the 'setting' and calls 'next_frame' to process the next frame.
    # ----------------------------------------------------------------------------------

    def detections(self, boxes, confidences, classIDs):
        
        self._setting.add_detections(boxes, confidences, classIDs)
        self.next_frame()
            
    # ----------------------------------------------------------------------------------
    # This is the main loop for the flow_manager. This method is registered as a
    # listener to the video_decoder 'doer'.  Whenever the video_decoder reads a new
    # frame it calls this method.  This method checks that the size of the decoded
    # video is correct and then reads the frame (from a mmap file). Then, it will
    # either track the itens in the frame or make a new detection.  If it is time
    # to make new detections, a message is sent to the Neural Network to 'find_bboxes'
    # with method 'detections' as the callback for when all the detections are
    # finished.
    # ----------------------------------------------------------------------------------

    def process_frame(self, size):

        # There was an error reading the last frame, so just move on to the next
        # frame
        if size < (self.height * self.width * self.depth):
            self.next_frame()
            return
            
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
        
