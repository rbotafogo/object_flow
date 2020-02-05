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
from object_flow.util.mmap_frames import MmapFrames

#==========================================================================================
# A Tracker tracks itens in a frame
#==========================================================================================

class Tracker(Doer):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def __init__(self):
        
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        
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

    def __initialize__(self, id, header_size, tracker_type = 'dlib'):
        # this tracker id
        self.id = id
        self.header_size = header_size
        self.tracker_type = tracker_type
            
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # SERVICES
    
    # ----------------------------------------------------------------------------------
    # Returns the id of this partial tracker
    # ----------------------------------------------------------------------------------

    def get_id(self):
        return self.id

    # ----------------------------------------------------------------------------------
    # Starts a dlib tracker to track the object given by its bounding box. Receives
    # the 'video_name' as parameter and will keep track of all objects by camera.
    # ATTENTION: This method is not efficient, since it starts tracking a single
    # item.  Better to use 'tracks_list' bellow.
    # ----------------------------------------------------------------------------------

    def start_tracking(self, video_name, file_name, frame_index, width, height,
                       depth, item_id, startX, startY, endX, endY):

        frame = self._get_frame(video_name, file_name, frame_index, width, height,
                                depth)
        
        logging.info("started tracking for video %s item_id %d", video_name, item_id)

        # gets the correct list of video items.
        video_items = self.videos.get(video_name, {})
        
        tracker = self._start_dlib_tracker(
            frame, item.startX, item.startY, item.endX, item.endY)
        
        # add this dlib tracker to the list of tracked items by this tracker for the
        # specified video
        video_items.update({item_id:tracker})
        self.videos[video_name] = video_items
        
    # ----------------------------------------------------------------------------------
    # Starts tracking a list of items in a video frame.  This is the preferred way of
    # starting trackers instead of the function above (start_tracking) that only
    # tracks 1 item in the video.
    # ----------------------------------------------------------------------------------

    def tracks_list(self, video_name, file_name, frame_index, width, height, depth,
                    items):
        
        frame = self._get_frame(video_name, file_name, frame_index, width, height,
                                depth)
        
        # gets the correct list of video items.
        video_items = self.videos.get(video_name, {})
        
        for item in items:
            # add this dlib tracker to the list of tracked items by this tracker for the
            # specified video
            tracker = self._start_tracker(
                frame, item.startX, item.startY, item.endX, item.endY)
            video_items.update({item.item_id:tracker})
            self.videos[video_name] = video_items

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def update_tracked_items(self, video_name, file_name, frame_index, width,
                             height, depth):

        frame = self._get_frame(video_name, file_name, frame_index, width, height,
                                depth)
        
        # get all tracked objects from the given camera
        if not (video_name in self.videos.keys()):
            return None

        video_items = self.videos[video_name]
        detections = {}
        
        for item_id, tracker in video_items.items():
            confidence, pos = self._update_tracker(frame, tracker, width, height)
            detections[item_id] = (confidence, pos)
            
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

    # PRIVATE METHODS
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _get_frame(self, video_name, file_name, frame_index, width, height, depth):

        size = width * height * depth
        
        # open the file descriptor if not already opened
        if not video_name in self._fd:
            self._fd[video_name] = os.open(file_name, os.O_RDONLY)

        # read the image
        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        # npage = math.ceil((width * height * depth)/ 4000) + 10
        npage = (math.ceil((width * height * depth)/ 4000) + 10) * (frame_index + 1)
        
        self._buf = mmap.mmap(
            self._fd[video_name], mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)
        self._buf.seek(frame_index * (size + self.header_size))
        # read the header
        self._buf.read(self.header_size)
        # read the frame
        b2 = np.frombuffer(self._buf.read(size), dtype=np.uint8)
        frame = b2.reshape((height, width, depth))

        return frame
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _get_frame2(self, video_name, file_name, frame_index, width, height, depth):

        size = width * height * depth
        
        # open the file descriptor if not already opened
        if not video_name in self._fd:
            self._fd[video_name] = os.open(file_name, os.O_RDONLY)

        # read the image
        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        # npage = math.ceil((width * height * depth)/ 4000) + 10
        npage = (math.ceil((width * height * depth)/ 4000) + 10) * (frame_index + 1)
        
        self._buf = mmap.mmap(
            self._fd[video_name], mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)
        self._buf.seek(frame_index * (size + self.header_size))
        # read the header
        self._buf.read(self.header_size)
        # read the frame
        b2 = np.frombuffer(self._buf.read(size), dtype=np.uint8)
        frame = b2.reshape((height, width, depth))

        return frame
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _start_tracker(self, frame, startX, startY, endX, endY):

        if self.tracker_type == 'dlib':
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(frame, rect)
        elif self.tracker_type in self.OPENCV_OBJECT_TRACKERS:
            tracker = self.OPENCV_OBJECT_TRACKERS[self.tracker_type]()
            tracker.init(frame, (startX, startY, endX - startX, endY - startY))
        else:
            logging.warning("Unknown tracker type: %s", self.tracker_type)
            # TODO: shutdown the system... no way to recover from that
        
        return tracker
                
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _update_tracker(self, frame, tracker, width, height):

        if self.tracker_type == 'dlib':
            confidence = tracker.update(frame)
            position = tracker.get_position()
            
            # make sure that the values are positive integers in the range (0, 0)
            # (width, height)
            pl = int(position.left())
            if pl < 0:
                pl = 0
                
            pt = int(position.top())
            if pt < 0:
                pt = 0
                
            pr = int(position.right())
            if pr > width:
                pr = width
                
            pb = int(position.bottom())
            if pb > height:
                pb = height

            position = [pl, pt, pr, pb]
        # if not a dlib tracker, it is a cv2 tracker
        else:
            confidence, position = tracker.update(frame)
            position = (position[0], position[1], position[0] + position[2],
                        position[1] + position[3])

        return (confidence, [int(value) for value in position])
