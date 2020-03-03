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
        # size in bytes of one bbox
        # Yolo: four integers, each with 4 bytes + 1 float for confidences (8 bytes),
        # + 1 int for classID (2 bytes)
        self.yolo_block_size = 4 * 4 + 1 * 8 + 1 * 2
        # Tracker: 1 bit for termination (1 byte) = 1 * 1 + 4 integers * 4 bytes for
        # bounding boxes 
        self.tracker_block_size = 1 * 1 + 4 * 4
        
        self.max_bboxes = 50
        # header has the number of of bounding boxes stored in the buffer
        # 1 integer
        self.header_size = 1 * 4

        # maximum size in bytes of bounding boxes for one video
        self.bboxes_size = self.header_size + self.max_bboxes * self.yolo_block_size

        self._alloc = mmap.ALLOCATIONGRANULARITY
        
    # ---------------------------------------------------------------------------------
    # Open mmap file for writing
    # ---------------------------------------------------------------------------------
 
    def create(self):
        self._fd = os.open(self.mmap_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC)
        os.write(self._fd, b'\x00' * mmap.ALLOCATIONGRANULARITY)
        
    # ---------------------------------------------------------------------------------
    # Open mmap file for writing. Assumes that the file was already created
    # ---------------------------------------------------------------------------------

    def open_write(self, video_name, video_id):
        
        self._fd = os.open(self.mmap_path, os.O_RDWR)
        # pg_length = self.bboxes_size * (video_id + 1)
        alloc = (self.bboxes_size * video_id) % self._alloc
        self._alloc = (alloc + 1) * mmap.ALLOCATIONGRANULARITY
        
        return mmap.mmap(self._fd, mmap.ALLOCATIONGRANULARITY,
                         access = mmap.ACCESS_WRITE,
                         offset = alloc * mmap.ALLOCATIONGRANULARITY)
    

        # It seems that there is no way to share memory between processes in
        # Windows, so we use mmap.ACCESS_WRITE that will store the frame on
        # the file. I had hoped that we could share memory.  In Linux, documentation
        # says that memory sharing is possible
        # return mmap.mmap(self._fd, 0, access = mmap.ACCESS_WRITE)
    
    # ---------------------------------------------------------------------------------
    # Open mmap file for reading only
    # ---------------------------------------------------------------------------------

    def open_read(self, video_name, video_id):

        # open the mmap file whith the decoded frame. 
        # number of pages is calculated from the image size
        # ceil((width x height x 3) / 4k (page size) + k), where k is a small
        # value to make sure that all image overhead are accounted for. 
        # pg_length = self.bboxes_size * (video_id + 1)
        self._fd = os.open(self.mmap_path, os.O_RDONLY)
        
        return mmap.mmap(self._fd, 0, access = mmap.ACCESS_READ)
        
    # ---------------------------------------------------------------------------------
    # Closes the mmap object
    # ---------------------------------------------------------------------------------

    def close(self, buf):
        buf.close()
        
    # ---------------------------------------------------------------------------------
    # read the header and advance the pointer in the file to the next byte
    # ---------------------------------------------------------------------------------

    def set_base_address(self, buf, video_id):
        pos = (video_id * self.bboxes_size)
        logging.debug("base position set to %d", pos)
        
        buf.seek(pos)

    # ---------------------------------------------------------------------------------
    # read the header and advance the pointer in the file to the next byte
    # ---------------------------------------------------------------------------------

    def set_detection_address(self, buf, video_id, box_index = 0):
        pos = ((video_id * self.bboxes_size) + # base address
               self.header_size +              # header
               (box_index * self.yolo_block_size))  # number of bounding boxes
        logging.debug("detection position set to %d", pos)
        
        buf.seek(pos)

    # ---------------------------------------------------------------------------------
    # reads the header and frame at the given index from the mmap file
    # ---------------------------------------------------------------------------------

    def read_data(self, buf, num_elmts, dtype):
        return np.frombuffer(
            buf.read(num_elmts * np.dtype(dtype).itemsize), dtype=dtype)
    
    # ---------------------------------------------------------------------------------
    # Write 0 to actually mapped file in memory
    # ---------------------------------------------------------------------------------

    def set0(self, buf):
        # os.write(self._fd, b'\x00' * mmap.PAGESIZE * self._npage)
        buf.write(b'\x00' * self.bboxes_size)
        
    # ---------------------------------------------------------------------------------
    # Write header
    # ---------------------------------------------------------------------------------

    def write_header(self, buf, num_elmts):
        size = buf.write(num_elmts)
    
    # ---------------------------------------------------------------------------------
    # Write the bounding box information. When migration to Cython, bbox is an
    # array of 4 ints of type int32, confidence is one float of type float and
    # classID is one int of type uint16
    # ---------------------------------------------------------------------------------

    def write_detection(self, buf, bbox, confidence, classID):
        size = buf.write(bbox)
        size += buf.write(confidence)
        size += buf.write(classID)
        logging.debug("writing box %s of size %d", bbox, size)
        return size
