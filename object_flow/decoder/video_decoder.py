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

        # number of frames read
        self.frame_number = 0
        
        # list of listeners interested to get a message everytime a new frame is
        # loaded
        self._listeners = {}

        self._frame_number_buffer = collections.deque()
        self._buffer_front = 0
        self._buffer_rear = 0
        self._drop_frames = False
                
        self._stream = None
        self._capture_average = None
        
        # TODO: filter initialization should be done in another way... This does not
        # allow for channing filters which would be ideal
        self._adjust_gamma = False
                
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __initialize__(self, video_name, path, buffer_max_size, header_size, width=500):
        
        self.path = path
        self.video_name = video_name
        self._buffer_max_size = buffer_max_size
        self.scaled_width = width
        
        # open a file for storing the frames
        self.file_name = "log/mmap_" + self.video_name
        self._fd = os.open(self.file_name, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
        
        # initialize the time counter
        self.init_time = time.perf_counter()

        # open the video file.  This will read and resize the image creating variables
        # self.width, self.height and self.depth
        self._open()
        self.frame_size = self.height * self.width * self.depth
        self.header_size = header_size
                        
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        # os.write(self._fd, b'\x00' * mmap.PAGESIZE * npage)
        npage = ((math.ceil((self.width * self.height * self.depth)/ 4000) + 10) *
                 self._buffer_max_size)
        os.write(self._fd, b'\x00' * mmap.PAGESIZE * npage)
        
        # It seems that there is no way to share memory between processes in
        # Windows, so we use mmap.ACCESS_WRITE that will store the frame on
        # the file. I had hoped that we could share memory.  In Linux, documentation
        # says that memory sharing is possible
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * npage,
                              access = mmap.ACCESS_WRITE)
        
        # start the drum_beat process
        self._drum_beat_address = self.hire(
            'DrumBeat', DrumBeat, self.video_name, timedelta(milliseconds=30),
            group = 'drum_beat')
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __hired__(self, hiree_name, hiree_group, hiree_address):
        if hiree_group == 'drum_beat':
            logging.info("%s: Drum beat hired", self.video_name)

            # starts the drum_beat.  DrumBeat call 'capture_next_frame' for
            # every listener
            self.post(self._drum_beat_address, 'add_listener', self.video_name,
                      self.myAddress)
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # SERVICES

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def get_image_info(self):
        return (self.file_name, self.width, self.height, self.depth)
    
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

            if self.frame_number % 100 == 0:
                now = time.perf_counter()
                self._capture_average = (now - self.init_time) / 100
                logging.debug("%s: buffer size is %d", self.video_name, len(self._frame_number_buffer))
                logging.debug("%s: average time video capture per frame for the last 100 frames is: %f",
                             self.video_name, self._capture_average)
                self.init_time = now

            # buffer not full yet... add frame to buffer
            if (len(self._frame_number_buffer) < self._buffer_max_size):
                if (self._drop_frames):
                    if ((self.frame_number % self._drop_by) != 0):
                        logging.debug("%s: adding frame %d", self.video_name,
                                     self.frame_number)
                        self._add_to_mmap(frame)
                else:
                    self._add_to_mmap(frame)
                
    # ----------------------------------------------------------------------------------
    # This method is called by the flow_manager to get the next available frame. Only
    # flow_manager should call this function.  Flow manager is registered as one of
    # the listeners to this decoder and this is how all frames are processed,
    # video_decoder next_frame and flow_manager's _next_frame each call each other
    # 'recursively'.
    # ----------------------------------------------------------------------------------

    def provide_next_frame(self, processing_average):

        # if mmap buffer empty, capture the next frame
        if  self._buffer_front == self._buffer_rear:
            self.capture_next_frame()

        # manage how fast we allow the buffer to grow based on the speed we are
        # consuming the buffer
        self._manage_buffer(processing_average)

        # return the frame_number and index of the next frame in the mmap file
        return self._get_next_mmap()
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS

    # ----------------------------------------------------------------------------------
    # adds a frame to the rear of the mmap file
    # ----------------------------------------------------------------------------------

    def _add_to_mmap(self, frame):
        
        next_index = self._buffer_rear + 1
        if next_index == self._buffer_max_size - 1:
            next_index = 0

        if next_index == self._buffer_front:
            logging.warning("%s: mmap buffer is full", self.video_name)
        else:
            self._buffer_rear = next_index
            # write the frame to the mmap file.  First move the offset to
            # correct position
            logging.debug("%s: writting to mmap position %d", self.video_name,
                          self._buffer_rear)
            self._buf.seek(self._buffer_rear * (self.frame_size + 1))
            self._buf.write(b'\x01')
            size = self._buf.write(frame)
            self._frame_number_buffer.append((self.frame_number, size))
            
    # ----------------------------------------------------------------------------------
    # 'remove' frame from mmap file
    # ----------------------------------------------------------------------------------

    def _get_next_mmap(self):
        if self._buffer_front == self._buffer_rear:
            logging.info("%s: mmap file is empty", self.video_name)
        else:
            if self._buffer_front == self._buffer_max_size - 1:
                self._buffer_front = 0
            else:
                self._buffer_front += 1

    # ----------------------------------------------------------------------------------
    # 'remove' frame from mmap file
    # ----------------------------------------------------------------------------------

    def _get_next_mmap2(self):
        if self._buffer_front == self._buffer_rear:
            logging.info("%s: mmap file is empty", self.video_name)
            self.capture_next_frame()
        else:
            frame_number, size = self._frame_number_buffer.popleft()
            if self._buffer_front == self._buffer_max_size - 1:
                self._buffer_front = 0
            else:
                self._buffer_front += 1

        return (frame_number, size, self._buffer_front)
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _manage_buffer(self, processing_average):
        if (processing_average != None and self._capture_average != None):
            per_diff = int(math.ceil(processing_average / self._capture_average))
            logging.debug("%s: speed difference is %d", self.video_name,
                          per_diff)
            if per_diff > 2:
                self._drop_frames = True
                self._drop_by = per_diff
                
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

