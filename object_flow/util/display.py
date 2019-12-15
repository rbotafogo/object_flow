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
        
        fd = os.open(mmap_path, os.O_RDONLY)
        self._buf = mmap.mmap(fd, 256 * mmap.PAGESIZE, access = mmap.ACCESS_READ)
        
    # ----------------------------------------------------------------------------------
    # Callback method called whenever a new frame is available in the mmap file
    # ----------------------------------------------------------------------------------

    def display(self, size):
        # logging.info("%s, %s, %s, display for video %s with size %d",
        #              Util.br_time(), os.getpid(), 'Display', video_name, size)

        if self._stop:
            return
        
        self._buf.seek(0)
        b2 = np.frombuffer(self._buf.read(size), dtype=np.uint8)
        frame = b2.reshape((self.height, self.width, self.depth))  # 480, 704, 3
        
        cv2.imshow("Iris 8 - Contagem - " + self.video_name, frame)
        cv2.waitKey(25)

    # ----------------------------------------------------------------------------------
    # Destroys the display window. Need to set self._stop = True to make sure that no
    # other frame will be shown, which would make the window reapear.
    # ----------------------------------------------------------------------------------

    def destroy_window(self):
        self._stop = True
        cv2.destroyAllWindows()        
        
