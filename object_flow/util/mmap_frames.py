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

import numpy as np

import logging

class MmapFrames:

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def __init__(self, video_name, width, height, depth):

        self.video_name = video_name
        self.mmap_path = "log/mmap_" + self.video_name
        self.width = width
        self.height = height
        self.depth = depth
        self.frame_size = width * height * depth
        
        self.buffer_max_size = 500
        self.page_size = 4000
        self.header_size = 4

    # ---------------------------------------------------------------------------------
    # Open mmap file for reading only
    # ---------------------------------------------------------------------------------

    def open_read(self):

        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        npage = ((math.ceil(self.frame_size / self.page_size) + 10) *
                 self.buffer_max_size)
        self._fd = os.open(self.mmap_path, os.O_RDONLY)
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)
        
    # ---------------------------------------------------------------------------------
    # Open mmap file for writing
    # ---------------------------------------------------------------------------------

    def open_write(self):
        
        self._fd = os.open(self.mmap_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
        npage = ((math.ceil(self.frame_size / self.page_size) + 10) *
                 self.buffer_max_size)
        # It seems that there is no way to share memory between processes in
        # Windows, so we use mmap.ACCESS_WRITE that will store the frame on
        # the file. I had hoped that we could share memory.  In Linux, documentation
        # says that memory sharing is possible
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * npage,
                              access = mmap.ACCESS_WRITE)
    
    # ---------------------------------------------------------------------------------
    # Write 0 to the whole file
    # ---------------------------------------------------------------------------------

    
    # ---------------------------------------------------------------------------------
    # Closes the mmap object
    # ---------------------------------------------------------------------------------

    def close(self):
        self._buf.close()
        
    # ---------------------------------------------------------------------------------
    # read the header and advance the pointer in the file to the next byte
    # ---------------------------------------------------------------------------------

    def set_pointer(self, frame_index):
        self._buf.seek(frame_index * (self.frame_size + self.header_size))

    # ---------------------------------------------------------------------------------
    # read the header and advance the pointer in the file to the next byte
    # ---------------------------------------------------------------------------------

    def read_header(self, frame_index):
        self.set_pointer(frame_size)
        return self._buf.read(self.header_size)

    # ---------------------------------------------------------------------------------
    # reads the header and frame at the given index from the mmap file
    # ---------------------------------------------------------------------------------

    def read_data(self, frame_index):
        
        self.set_pointer(frame_index)
        header = self._buf.read(self.header_size)
        b2 = np.frombuffer(self._buf.read(self.frame_size), dtype=np.uint8)
        frame = b2.reshape((self.height, self.width, self.depth))

        return (header, frame)
    
