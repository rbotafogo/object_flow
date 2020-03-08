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

import logging
import time

import cv2
import numpy as np

# tracking algorithm
import dlib

from object_flow.ipc.doer import Doer
from object_flow.util.stopwatch import Stopwatch

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

        # number of frames processed by this tracker
        self._total_frames = 0
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __initialize__(self, id, tracker_type = 'dlib'):
        # this tracker id
        self.id = id
        self.tracker_type = tracker_type
            
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
    # Returns the id of this partial tracker
    # ----------------------------------------------------------------------------------

    def get_id(self):
        return self.id

    # ----------------------------------------------------------------------------------
    # Registers a video with this tracker
    # ----------------------------------------------------------------------------------

    def register_video(self, video_name, video_id, width, height, depth):
        self.videos[video_name] = {}
        self.videos[video_name]['items'] = {}
        self.videos[video_name]['video_id'] = video_id
        self.videos[video_name]['width'] = width
        self.videos[video_name]['height'] = height
        self.videos[video_name]['depth'] = depth
        self.videos[video_name]['frame_size'] = width * height * depth
        self.videos[video_name]['frames'] = MmapFrames(video_name, width, height, depth)
        self.videos[video_name]['frames'].open_read()
        
    # ----------------------------------------------------------------------------------
    # Starts tracking a list of items in a video frame.  This is the preferred way of
    # starting trackers instead of the function above (start_tracking) that only
    # tracks 1 item in the video.
    # ----------------------------------------------------------------------------------

    def tracks_list(self, video_name, frame_index, items):
        
        frame = self._get_frame(video_name, frame_index)
        
        # gets the correct list of video items.
        video_items = self.videos.get(video_name)['items']
        
        for item in items:
            # add this dlib tracker to the list of tracked items by this tracker for the
            # specified video
            tracker = self._start_tracker(
                frame, item.startX, item.startY, item.endX, item.endY)
            video_items.update({item.item_id:tracker})
            
            self.videos[video_name]['items'] = video_items

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def update_tracked_items(self, video_name, frame_index):

        Stopwatch.start('update_tracking')
        self._total_frames += 1
        logging.debug("%s: number of tracked items is %d", str(self.id),
                      len(self.videos[video_name]['items']))
        
        if len(self.videos[video_name]['items']) == 0:
            return None
        
        frame = self._get_frame(video_name, frame_index)
        width = self.videos[video_name]['width']
        height = self.videos[video_name]['height']
        
        video_items = self.videos[video_name]['items']
        detections = {}
        logging.info("Tracker%s, Item length:%s", self.id, len(video_items))
        for item_id, tracker in video_items.items():
            start_time = time.perf_counter()
            confidence, pos = self._update_tracker(frame, tracker, width, height)
            if (any(x < 0 for x in pos) or any (x > 65535 for x in pos)):
                confidence = -1
                pos = [0, 0, 0, 0]
            else:
                pos = np.array(pos, dtype=np.uint16)
                
            detections[item_id] = (confidence, pos)
            end_time = time.perf_counter()
            # logging.info("Tracker Id: %s, Video name: %s, Item id: %d, Time took: %f",str(self.id), video_name, item_id, end_time-start_time)

        Stopwatch.stop('update_tracking') 
        Stopwatch.report(str(self.id), self._total_frames)       
        
        return detections

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def stop_tracking(self, video_name, item_id):
        video_items = self.videos[video_name]['items']
        del video_items[item_id]
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def stop_tracking_items(self, video_name, items_ids):
        for item_id in items_ids:
            del self.videos[video_name]['items'][item_id]
            
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

    def _get_frame(self, video_name, frame_index):

        header, frame = self.videos[video_name]['frames'].read_data(frame_index)
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
