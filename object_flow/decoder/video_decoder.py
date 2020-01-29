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
import collections
from urllib.parse import urlparse
import time
from datetime import timedelta

import cv2
import logging
import numpy as np
from imutils.video import FPS
import imutils

from object_flow.ipc.doer import Doer
from object_flow.decoder.drum_beat import DrumBeat

# from datetime import timedelta
# from sys import getsizeof
# CHECK_PERIOD = timedelta(milliseconds=25)

#==========================================================================================
#
#==========================================================================================

class VideoDecoder(Doer):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        
        self.frame_number = 0
        self._not_grabbed = 0
        
        # list of listeners interested to get a message everytime a new frame is
        # loaded
        self._listeners = {}

        # frame_buffer
        self._frame_buffer = collections.deque()
        
        # Maximum size of the frame buffer
        self._buffer_max_size = 500
        
        self._stream = None
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __initialize__(self, video_name, path, width = 500):
        self.path = path
        self.video_name = video_name
        self.scaled_width = width
        self.init_time = time.perf_counter()

        # start the drum_beat process
        self._drum_beat_address = self.hire(
            'DrumBeat', DrumBeat, self.video_name, timedelta(milliseconds=30),
            group = 'drum_beat')

        # TODO: filter initialization should be done in another way... This does not
        # allow for channing filters which would be ideal
        self._adjust_gamma = False
        
        # open the video file
        self._open()
        
        # open a file for storing the frames
        self.file_name = "log/mmap_" + self.video_name
        self._fd = os.open(self.file_name, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
        
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        # os.write(self._fd, b'\x00' * mmap.PAGESIZE * npage)
        npage = math.ceil((self.width * self.height * self.depth)/ 4000) + 10
        os.write(self._fd, b'\x00' * mmap.PAGESIZE * npage)
        
        # It seems that there is no way to share memory between processes in
        # Windows, so we use mmap.ACCESS_WRITE that will store the frame on
        # the file. I had hoped that we could share memory.  In Linux, documentation
        # says that memory sharing is possible
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * npage,
                              access = mmap.ACCESS_WRITE)

        self.post(self._drum_beat_address, 'add_listener', self.video_name,
                  self.myAddress)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __hired__(self, hiree_name, hiree_group, hiree_address):
        if hiree_group == 'drum_beat':
            logging.info("%s: Drum beat hired", self.video_name)
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # SERVICES

    # ----------------------------------------------------------------------------------
    # Adds a new listener to this decoder. When a new listener is added it can receive
    # use the values of width, height and depth already initialized from the camera
    # ----------------------------------------------------------------------------------

    def add_listener(self, name, address, callback):
        self._listeners[name] = (address, callback)
        return (self.file_name, self.width, self.height, self.depth)
    
    # ----------------------------------------------------------------------------------
    # Removes the listener
    # ----------------------------------------------------------------------------------

    def remove_listener(self, name):
        del self._listeners[name]
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def gamma_filter(self, value):
        # table to adjust the pixel's gamma value
        self._adjust_gamma = True
        self._define_gamma_table(value)
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def add_filter(self, filter_name, *args, **kwargs):
        method = getattr(self, filter_name)
        method(*args, **kwargs)
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # PROTECTED METHODS

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def start_processing(self):
        # start the frames per second throughput estimator
        self._fps = FPS().start()
        self.init_time = time.perf_counter()
        self.next_frame()
        
    # ----------------------------------------------------------------------------------
    # This method is called by drum_beat.  DrumBeat conducts the capturing of frames.
    # When working with video files, DrumBeat will delay frame capturing to the
    # processing rate. With life files, DrumBeat will beat at 30 frames per second.
    # ----------------------------------------------------------------------------------

    def capture_next_frame(self):
        # (grabbed, frame) = self._stream.read()
        grabbed = self._stream.grab()
        (grabbed, frame) = self._stream.retrieve()
        self.frame_number += 1
       
        if not grabbed:
            logging.warning("%s: could not grab video stream", self.video_name)
            self._open()
        else:
            # resize image
            frame = cv2.resize(frame, self.dim, interpolation = cv2.INTER_AREA)
            
            if self._adjust_gamma:
                frame = cv2.LUT(frame, self._gamma_table)

            # self._manage_buffer()

            if self.frame_number % 100 == 0:
                now = time.perf_counter()
                logging.info("%s: buffer size is %d", self.video_name, len(self._frame_buffer))
                logging.info("%s: average time video capture per frame for the last 100 frames is: %f",
                             self.video_name, (now - self.init_time) / 100)
                self.init_time = now

            # buffer not full yet... add frame to buffer
            if len(self._frame_buffer) < self._buffer_max_size:
                self._frame_buffer.append((self.frame_number, frame))
            # buffer is full
            else:
                pass
                # logging.warning("%s - frame buffer overflow", self.video_name)
                # reduce the size of the buffer gracefully accross the whole buffer
                # self._del_buffer_every(5)

    # ----------------------------------------------------------------------------------
    # This method is called by the flow_manager to get the next available frame. Only
    # flow_manager should call this function.  Flow manager is registered as one of
    # the listeners to this decoder and this is how all frames are processed,
    # video_decoder next_frame and flow_manager's _next_frame each call each other
    # 'recursively'.
    # ----------------------------------------------------------------------------------

    def next_frame(self):

        if (len(self._frame_buffer) == 0):
            self.capture_next_frame()
            
        frame_number, frame = self._frame_buffer.popleft()
        
        # write the frame to the mmap file.  First move the offset to
        # position 0
        self._buf.seek(0)
        tot = self._buf.write(frame)
        
        for name, listener in self._listeners.items():
            # listener[0]: doer's address
            # listener[1]: doer's method to call
            self.post(listener[0], listener[1], tot, frame_number)  
                
        self._fps.update()
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _live_cam(self):
        # consuming the buffer to fast? We've already consumed half of the
        # buffer size... start dropping frames
        if (len(self._frame_buffer) >
            (self._buffer_max_size - self._first_measure) / 2):
            self._drop_frames = True
            if self._drop_frames_by > 3:
                self._drop_frames_by -= 1
                self._del_buffer_every(self._drop_frames)
                logging.info("%s: increase dropping frames rate to %d", self.video_name,
                             self._drop_frames)
        # consuming the buffer to slowly? throw away less frames
        if (len(self._frame_buffer) < (self._first_measure - 0) / 2):
            if (self._drop_frames_by < 8):
                self._drop_frames_by += 1
            else:
                self._drop_frames = False
                logging.info("%s: decrese dropping frames rate to %d", self.video_name,
                             self._drop_frames)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS

    # ----------------------------------------------------------------------------------
    # Table to adjust the image gamma filter
    # ----------------------------------------------------------------------------------

    def _define_gamma_table(self, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        self._gamma_table = np.array([((i / 255.0) ** invGamma) * 255
                                      for i in np.arange(0, 256)]).astype("uint8")

    
    # ----------------------------------------------------------------------------------
    # When using self.wakeupAfter the process does not receive any other messages.
    # This is not a good solution, at least not in Windows.  Haven't checked it in
    # Linux
    # ----------------------------------------------------------------------------------

    def _wakeup(self):
        self._next_frame()
        self.wakeupAfter(CHECK_PERIOD)
        
    # ----------------------------------------------------------------------------------
    # Deletes every 'n' frames from the buffer, so that we degrade gracefully the
    # quality
    # ----------------------------------------------------------------------------------

    def _del_buffer_every(self, n):
        for i in range(len(self._frame_buffer) -1, 1, -n):
            del self._frame_buffer[i]
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _open(self):
        # already opened, but failed for some reason.  Close it and open again
        if self._stream != None and self._stream.isOpened():
            self._stream.release()

        # check if the path has a schema such as 'rtsp', if if does, this is a
        # live_cam and we cannot reduce the drum_beat.  If not a live_cam, the
        # we should reduce the drum_beat to match the processing speed.
        url_parse = urlparse(self.path)
        if (url_parse.scheme == ''):
            self.live_cam = False
        else:
            self.live_cam = True
            
        self._stream = cv2.VideoCapture(self.path)

        if not self._stream.isOpened():
            logging.warning("Could not open video stream %s on path %s", video_name, self.path)
        else:
            logging.info("Starting decoding video %s in path %s", self.video_name, self.path)
            
            # Read all the video properties        
            self._read_properties()
            
            # Scale the image
            scale_percent = (self.scaled_width * 100) / self.width
            self.width = int(self.width * scale_percent / 100)
            self.height = int(self.height * scale_percent / 100)
            self.dim = (self.width, self.height)
            
            logging.info("%s: width: %d, height: %d, fps: %f", self.video_name, self.width,
                         self.height, self.fps)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _read_properties(self):
        self.width  = int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
        self.height = int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
        self.depth = 3   # default value... don't know it this is always the case
        self.fps = self._stream.get(cv2.CAP_PROP_FPS)
        self.format = self._stream.get(cv2.CAP_PROP_FORMAT)
        self.fourcc = self._stream.get(cv2.CAP_PROP_FOURCC)
        self.frame_count = self._stream.get(cv2.CAP_PROP_FRAME_COUNT)
        self.brightness = self._stream.get(cv2.CAP_PROP_BRIGHTNESS)
        self.contrast = self._stream.get(cv2.CAP_PROP_CONTRAST)
        self.saturation = self._stream.get(cv2.CAP_PROP_SATURATION)
        self.hue = self._stream.get(cv2.CAP_PROP_HUE)
        self.gain = self._stream.get(cv2.CAP_PROP_GAIN)
        self.exposure = self._stream.get(cv2.CAP_PROP_EXPOSURE)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _check_frames(self):
        # try to determine the total number of frames in the video file
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            self.total = int(self._stream.get(prop))
            logging.info("%s: total frames in video %d",
                         self.camera_id, self.total)
        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            self.total = -1

