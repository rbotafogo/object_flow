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

import cv2
import numpy as np

# tracking algorithm
import dlib

from object_flow.ipc.doer import Doer
from object_flow.flow.item import Item
from object_flow.flow.setting import Setting

#==========================================================================================
# A Tracker tracks itens in a frame
#==========================================================================================

class Tracker(Doer):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def __init__(self):
        # every Doer should call super().__init__() if it has an __init__ method
        super().__init__()

        # A Tracker can track itens from different videos, so it has a list of all
        # videos and for each video, the list of itens it is tracking
        self.videos = {}
        
        # file descriptors opened where frames are stored
        self._fd = {}
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __initialize__(self, id):
        # this tracker id
        self.id = id
        
    # ----------------------------------------------------------------------------------
    # Returns the id of this partial tracker
    # ----------------------------------------------------------------------------------

    def get_id(self):
        return self.id

    # ----------------------------------------------------------------------------------
    # Starts a dlib tracker to track the object given by its bounding box. Receives
    # the 'video_analyser' as parameter and will keep track of all objects by camera.
    # _start_tracker is started when the process receives a 'Start' message.
    # TODO: uses explicitly the dlib tracker.  Should configure so that another tracker
    # could be used.
    # ----------------------------------------------------------------------------------

    def start_tracking(self, video_name, file_name, width, height, depth, size,
                       item_id, startX, startY, endX, endY):

        frame = self._get_frame(video_name, file_name, width, height, depth, size)
        
        logging.info("started tracking for video %s item_id %d", video_name, item_id)

        # gets the correct list of video items.
        video_items = self.videos.get(video_name, {})
        
        dlib_tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        dlib_tracker.start_track(frame, rect)
        
        # add this dlib tracker to the list of tracked items by this tracker for the
        # specified video
        video_items.update({item_id:dlib_tracker})
        self.videos[video_name] = video_items
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def update_tracked_items(self, video_name, file_name, width, height, depth, size):

        frame = self._get_frame(video_name, file_name, width, height, depth, size)
        
        # get all tracked objects from the given camera
        if not (video_name in self.videos.keys()):
            return None

        video_items = self.videos[video_name]

        detections = {}
        
        for item_id, dlib_tracker in video_items.items():
            
            confidence = dlib_tracker.update(frame)
            pos = dlib_tracker.get_position()
            
            # make sure that the values are positive integers in the range (0, 0)
            # (width, height)
            pl = int(pos.left())
            if pl < 0:
                pl = 0
                
            pt = int(pos.top())
            if pt < 0:
                pt = 0
                
            pr = int(pos.right())
            if pr > width:
                pr = width
                
            pb = int(pos.bottom())
            if pb > height:
                pb = height

            detections[item_id] = (confidence, [pl, pt, pr, pb])
    
        return detections

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def stop_tracking(self, video_name, item_id):
        video_items = self.videos[video_name]
        del video_items[item_id]
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def say_hello(self, *args, **kwargs):
        logging.info("Hello from tracker %d with args %s and kwargs %s", self.id,
                     args, kwargs)
        

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _get_frame(self, video_name, file_name, width, height, depth, size):

        # open the file descriptor if not already opened
        if not video_name in self._fd:
            self._fd[video_name] = os.open(file_name, os.O_RDONLY)

        # read the image
        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        npage = math.ceil((width * height * depth)/ 4000) + 10
        
        self._buf = mmap.mmap(
            self._fd[video_name], mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)
        self._buf.seek(0)
        b2 = np.frombuffer(self._buf.read(size), dtype=np.uint8)
        frame = b2.reshape((height, width, depth))

        return frame
