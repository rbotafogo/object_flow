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

    def __init__(self, mmap_path, width, height, depth):

        self.mmap_path = mmap_path
        self.width = width
        self.height = height
        self.depth = depth
        self.frame_size = width * height * depth
        
        self.buffer_max_size = 500
        self.page_size = 4000
        self.header_size = 4

    # ---------------------------------------------------------------------------------
    # Open mmap file
    # ---------------------------------------------------------------------------------

    def open_read(self):

        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        # npage = math.ceil((self.width * self.height * self.depth)/ 4000) + 10
        npage = ((math.ceil(self.frame_size / self.page_size) + 10) *
                 self.buffer_max_size)
        self._fd = os.open(self.mmap_path, os.O_RDONLY)
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * npage, access = mmap.ACCESS_READ)
        
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
    
