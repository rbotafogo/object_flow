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
        self.header_size = 8

        self._buffer_rear = 0
        
    # ---------------------------------------------------------------------------------
    # Open mmap file for reading only
    # ---------------------------------------------------------------------------------

    def open_read(self):

        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        self._npage = ((math.ceil(self.frame_size / self.page_size) + 10) *
                       self.buffer_max_size + 1)
        self._fd = os.open(self.mmap_path, os.O_RDONLY)
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * self._npage,
                              access = mmap.ACCESS_READ)
        
    # ---------------------------------------------------------------------------------
    # Open mmap file for writing
    # ---------------------------------------------------------------------------------

    def open_write(self):
        
        self._fd = os.open(self.mmap_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
        self._npage = ((math.ceil(self.frame_size / self.page_size) + 10) *
                       self.buffer_max_size + 1)
        # It seems that there is no way to share memory between processes in
        # Windows, so we use mmap.ACCESS_WRITE that will store the frame on
        # the file. I had hoped that we could share memory.  In Linux, documentation
        # says that memory sharing is possible
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * self._npage,
                              access = mmap.ACCESS_WRITE)
    
    # ---------------------------------------------------------------------------------
    # Open mmap file for writing
    # ---------------------------------------------------------------------------------

    def open_write2(self):
        
        self._fd = os.open(self.mmap_path, os.O_RDWR)
        self._npage = ((math.ceil(self.frame_size / self.page_size) + 10) *
                       self.buffer_max_size + 1)
        # It seems that there is no way to share memory between processes in
        # Windows, so we use mmap.ACCESS_WRITE that will store the frame on
        # the file. I had hoped that we could share memory.  In Linux, documentation
        # says that memory sharing is possible
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * self._npage,
                              access = mmap.ACCESS_WRITE)
    
    # ---------------------------------------------------------------------------------
    # Closes the mmap object
    # ---------------------------------------------------------------------------------

    def close(self):
        self._buf.close()
        
    # ---------------------------------------------------------------------------------
    # Write 0 to actually mapped file in memory
    # ---------------------------------------------------------------------------------

    def set0(self):
        os.write(self._fd, b'\x00' * mmap.PAGESIZE * self._npage)
        
    # ---------------------------------------------------------------------------------
    # read the header and advance the pointer in the file to the next byte
    # ---------------------------------------------------------------------------------

    def set_pointer(self, frame_index):
        self._buf.seek(frame_index * (self.frame_size + self.header_size))

    # ---------------------------------------------------------------------------------
    # read the header and advance the pointer in the file to the next byte
    # ---------------------------------------------------------------------------------

    def read_header(self, frame_index):
        self.set_pointer(frame_index)
        header = self._buf.read(self.header_size)
        frame_number = int.from_bytes(header, byteorder = 'big')
        
        return frame_number

    # ---------------------------------------------------------------------------------
    # reads the header and frame at the given index from the mmap file
    # ---------------------------------------------------------------------------------

    def read_data(self, frame_index):
        
        self.set_pointer(frame_index)
        header = self._buf.read(self.header_size)
        frame_number = int.from_bytes(header, byteorder = 'big')
        b2 = np.frombuffer(self._buf.read(self.frame_size), dtype=np.uint8)
        frame = b2.reshape((self.height, self.width, self.depth))

        return (frame_number, frame)
    
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def copy_last(self, frame_index):
        frame_number, frame = self.read_data(frame_index)
        self._write_frame(self.buffer_max_size, frame, frame_number)
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def read_last(self):
        return self.read_data(self.buffer_max_size)
    
    # ---------------------------------------------------------------------------------
    # Write the frame header
    # ---------------------------------------------------------------------------------

    def write_header(self, frame_index, value):
        val = value.to_bytes(8, byteorder = 'big')
        
        self.set_pointer(frame_index)
        self._buf.write(val)
    
    # ---------------------------------------------------------------------------------
    # Write the frame
    # ---------------------------------------------------------------------------------

    def _write_frame(self, index, frame, frame_number):
        self.write_header(index, frame_number)
        size = self._buf.write(frame)
        return size
        
    # ---------------------------------------------------------------------------------
    # Write the frame
    # ---------------------------------------------------------------------------------

    def write_frame(self, frame, frame_number):
        # fn = frame_number.to_bytes(8, byteorder = 'big')
        
        next_index = self._buffer_rear + 1
        if next_index == self.buffer_max_size - 1:
            next_index = 0

        # if next frame in the buffer has not yet been processed, then just drop
        # the frame
        val = self.read_header(next_index)
        if val != 0:
            return 0

        # check to see if the frame was already processed.  If not, wait to write
        # the frame.  This will block the video_decoder.
        # val = -1
        # while val != 0:
        #     val = self.read_header(next_index)

        # move last element of buffer to the next index
        self._buffer_rear = next_index
        
        logging.debug("%s: writting to mmap position %d", self.video_name,
                      self._buffer_rear)
        
        # write the frame to the mmap file.
        self.write_header(next_index, frame_number)
        size = self._buf.write(frame)
        return size
    
