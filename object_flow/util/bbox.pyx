 # -*- coding: utf-8 -*-

##########################################################################################
# @author Rodrigo Botafogo
#
# Copyright (C) 2019 Rodrigo Botafogo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019
##########################################################################################

import logging

cdef class cBBox:
    cdef:
        short startX
        short startY
        short endX
        short endY
        short class_id
        double confidence
        
    def __init__(self, np_bbox, short class_id, double confidence):
        self.startX = np_bbox[0]
        self.startY = np_bbox[1]
        self.endX = np_bbox[2]
        self.endY = np_bbox[3]
        self.class_id = class_id
        self.confidence = confidence

    def log(self):
        logging.info("bounding box (%d, %d, %d, %d) - class_id: %d - confidence %.2f",
                     self.startX, self.startY, self.endX, self.endY,
                     self.class_id, self.confidence)
