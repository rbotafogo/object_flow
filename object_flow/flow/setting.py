# -*- coding: utf-8 -*-

##########################################################################################
# @author Rodrigo Botafogo
#
# Copyright (C) 2019 Rodrigo Botafogo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019
##########################################################################################

import time
import itertools
import logging

from object_flow.flow.item import Item
from object_flow.util.geom import Geom

# =========================================================================================
# Class Setting is responsible for managing all the items in the camera view
# =========================================================================================

class Setting:

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def __init__(self, cfg):
        self.cfg = cfg

        # list of items in this setting
        self.items = []
        
    # ---------------------------------------------------------------------------------
    # Convert every new identified bounding box to an iten and returns the list of
    # new items.  Those items will not necessarily be added to the Setting.  The
    # identified bounding boxes could already be on the Setting or should not yet
    # be added to it.
    # ---------------------------------------------------------------------------------

    def bboxes2items(self, bboxes, confidences, class_ids):

        new_inputs = []
        
        for (i, bbox) in enumerate(bboxes):
            item = Item(bbox[0], bbox[1], bbox[2], bbox[3], class_ids[i],
                        confidences[i])
            new_inputs.append(item)

        return new_inputs

    # ---------------------------------------------------------------------------------
    # Only itens that have crossed the entry_lines (in the right direction) should be
    # considered for addition to the setting. If a bounding box is split by an entry
    # line, then it should not be added to the setting, only items that are completely
    # inside the entry lines should be considered
    # ---------------------------------------------------------------------------------

    def _validate_entry(self, bboxes):

        valid_boxes = []
        
        # for every entry line
        for box in bboxes:
            add_box = True
            
            for key, spec in self.cfg.data['entry_lines'].items():
                end_points = spec["end_points"]
                try: 
                    top = Geom.point_position(end_points[0], end_points[1],
                                              end_points[2], end_points[3],
                                              box[0], box[1])
                except OverflowError:
                    logging.info("overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                                 end_points[0], end_points[1],
                                 end_points[2], end_points[3],
                                 box[0], box[1], self.cfg.analyser_id)
                    top = False

                try: 
                    bottom = Geom.point_position(end_points[0], end_points[1],
                                                 end_points[2], end_points[3],
                                                 box[2], box[3])
                    
                except OverflowError:
                    logging.info("overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                                 end_points[0], end_points[1],
                                 end_points[2], end_points[3],
                                 box[2], box[3],  self.cfg.analyser_id)
                    bottom = False
                
                # split object: should not be added to the tracked objects
                if top != bottom:
                    add_box = False
                      
                # side1 indicates the valid region for entry, that is, the area
                # in which if the object´s bounding box is completely inside
                # then it should be seen.
                # This is not an split object, so we can check either the top
                # or bottom.  Accept if they are the same 
                if ((not top and spec['side1'] == 'Positive') or
                    (top and spec['side1'] == 'Negative')):
                    add_box = False
                    
            if add_box:
                valid_boxes.append(box)
                
        return valid_boxes
    
    # ---------------------------------------------------------------------------------
    # Add new detected elements to the setting if they are not already tracked and
    # returns the added elements as Items
    # ---------------------------------------------------------------------------------

    def add_detections(self, bboxes, confidences, class_ids):

        # Checks if the item should be added/removed from the Setting.  Items should
        # only be in the Setting if they are inside the entry lines.
        bboxes = self._validate_entry(bboxes)

        # convert the bounding boxes to items
        new_inputs = self.bboxes2items(bboxes, class_ids, confidences)

        # TODO: lots of things....!!!
        self.items = new_inputs
