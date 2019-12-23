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
import cv2
import logging
import mmap
import numpy as np
import math

from datetime import timedelta

from object_flow.util.util import Util
from object_flow.ipc.doer import Doer

#==========================================================================================
#
#==========================================================================================

class Display(Doer):

    # ----------------------------------------------------------------------------------
    # When Display is create the initialize method should be called first.
    # @param video_name [String] name of the video camera
    # ----------------------------------------------------------------------------------

    def initialize(self, video_name):
        self.video_name = video_name
        self._stop = False
    
    # ----------------------------------------------------------------------------------
    # Callback method needed to initialize the mmap file.  This could be called by
    # any mmap file generator.
    # @param mmap_path [String] path of the mmap file
    # @param width [Integer] width of the image in the mmap file
    # @param height [Integer] height of the image in the mmap file
    # @param depth [Integer] depth of the image in the mmap file
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

        self._buf = mmap.mmap(fd, mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def base_image(self, size):
        if self._stop:
            return
        
        self._buf.seek(0)
        b2 = np.frombuffer(self._buf.read(size), dtype=np.uint8)
        self.frame = b2.reshape((self.height, self.width, self.depth))  # 480, 704, 3

    # ----------------------------------------------------------------------------------
    # Callback method called whenever a new frame is available in the mmap file
    # ----------------------------------------------------------------------------------

    def display(self, size):
        # logging.debug("%s, %s, %s, display for video %s with size %d",
        #               Util.br_time(), os.getpid(), 'Display', video_name, size)
        cv2.imshow("Iris 8 - Contagem - " + self.video_name, self.frame)
        cv2.waitKey(25)

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    def overlay_bboxes(self, items):
        for item in items:
            # logging.info((item.startX, item.startY, item.endX, item.endY))
            cv2.rectangle(self.frame, (item.startX, item.startY),
                          (item.endX, item.endY), (0, 250, 0), 2)
    
    # ----------------------------------------------------------------------------------
    # Destroys the display window. Need to set self._stop = True to make sure that no
    # other frame will be shown, which would make the window reapear.
    # ----------------------------------------------------------------------------------

    def destroy_window(self):
        self._stop = True
        cv2.destroyAllWindows()        
        
