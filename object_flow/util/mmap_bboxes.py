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

class MmapBboxes:

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def __init__(self):

        self.mmap_path = "log/mmap_bboxes"
        self.page_size = 4000
        self.header_size = 8
        # size in bytes of one bounding box
        # four integers, each with 4 bytes + 1 float for confidences (4 bytes),
        # + 1 int for classID (4 bytes)
        self.bbox_size = 4 * 4 + 1 * 4 + 1 * 4
        self.max_bboxes = 50
        self.num_videos = 0

        # maximum size in bytes of bounding boxes for one video
        self.bboxes_size = self.max_bboxes * self.bbox_size

    # ---------------------------------------------------------------------------------
    # Open mmap file for reading only
    # ---------------------------------------------------------------------------------

    def open_read(self):

        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        self._npage = ((math.ceil(self.bboxes_size / self.page_size) + 10)
        self._fd = os.open(self.mmap_path, os.O_RDONLY)
        self._buf = mmap.mmap(self._fd, mmap.PAGESIZE * self._npage,
                              access = mmap.ACCESS_READ)
        
    # ---------------------------------------------------------------------------------
    # Open mmap file for writing
    # ---------------------------------------------------------------------------------

    def open_write(self):
        
        self._fd = os.open(self.mmap_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
        self._npage = (math.ceil(self.bboxes_size / self.page_size) + 10)
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
    # Write the frame
    # ---------------------------------------------------------------------------------

    def write_bbox(self, bbox, confidence, classID):
        next_index = self._buffer_rear + 1
        if next_index == self.buffer_max_size - 1:
            next_index = 0

        # check to see if the frame was already processed
        val = -1
        while val != 0:
            val = int.from_bytes(
                self.read_header(next_index), byteorder = 'big')
            # if val != 0:
            #     logging.info("Waiting for flow_manager to finish processing the frame")
            logging.debug("******index %d: buffer value %d ********",
                          next_index, val)
            
        self._buffer_rear = next_index
        # write the frame to the mmap file.  First move the offset to
        # correct position
        logging.debug("%s: writting to mmap position %d", self.video_name,
                      self._buffer_rear)
        # self._buf.seek(self._buffer_rear * (self.frame_size + self.header_size))
        # self._buf.write(fn)
        self.write_header(next_index, frame_number)
        size = self._buf.write(frame)
        return size
    
        
