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
# Class Setting is responsible for managing all the items in the camera view. This is
# an auxiliary class for flow_manager.
# =========================================================================================

class Setting:

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def __init__(self, cfg):
        self.cfg = cfg
        self._set_counters()

        # id of the next item
        self.next_item_id = 0
        
        # dictionary of items in this setting
        self.items = {}
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    # PROTECTED METHODS
    
    # ---------------------------------------------------------------------------------
    # Add new detected elements to the setting if they are not already tracked and
    # returns the added elements as Items
    # ---------------------------------------------------------------------------------

    def detections2items(self, bboxes, confidences, class_ids):

        # Checks if the item should be added/removed from the Setting.  Items should
        # only be in the Setting if they are inside the entry lines.
        bboxes = self._validate_entry(bboxes)

        # convert the bounding boxes to items
        self.new_inputs = self._bboxes2items(bboxes, class_ids, confidences)
        for item in self.new_inputs:
            # self._init_item_counters(item)
            for key in self.cfg.data['counting_lines']:
                item.init_lines(key, self.cfg.frame_number)

    # ---------------------------------------------------------------------------------
    # checks if the item has crossed an entry_line and is exiting the setting
    # ---------------------------------------------------------------------------------

    def check_exit(self, bounding_box):

        # for every entry line
        for key, spec in self.cfg.data["entry_lines"].items():
            end_points = spec["end_points"]
            try: 
                new_top = Geom.point_position(
                    end_points[0], end_points[1], end_points[2], end_points[3],
                    bounding_box[0], bounding_box[1])
            except OverflowError:
                logging.info(
                    "check_exit: new_top overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                    end_points[0], end_points[1], end_points[2], end_points[3],
                    bounding_box[0], bounding_box[1], self.video_name)
                new_top = False

            try:
                new_bottom = Geom.point_position(
                    end_points[0], end_points[1], end_points[2], end_points[3],
                    bounding_box[0], bounding_box[1])                    
            except OverflowError:
                logging.info(
                    "check_exit: new_bottom overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                    end_points[0], end_points[1], end_points[2], end_points[3],
                    bounding_box[2], bounding_box[3], self.video_name)
                new_bottom = False
                
            # side1 indicates the valid region for entry, that is, the area
            # in which if the object´s bounding box is completely inside
            # then it should be seen.
            # This is not an split object, so we can check either the new_top
            # or new_bottom.  Accept if they have the same 
            if ((new_top == new_bottom) and
                ((not new_top and spec['side1'] == 'Positive') or
                 (new_top and spec['side1'] == 'Negative'))):
                return True
                    
    # ---------------------------------------------------------------------------------
    # 
    # ---------------------------------------------------------------------------------

    def update_item(self, frame_number, item_id, confidence, bounding_box):
        logging.debug("updating item %d boundign box with confidence %f to: %s",
                      item_id, confidence, bounding_box)

        # item might have disappeared after tracking started
        if item_id not in self.items.keys():
            return
        
        self.items[item_id].tracker_update(
            frame_number, confidence, bounding_box[0], bounding_box[1],
            bounding_box[2], bounding_box[3])
    
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    # PRIVATE METHODS
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _set_counters(self):

        for key, spec in self.cfg.data["counting_lines"].items():

            logging.info("setting line %s for video %s", key, self.cfg.video_name)

            spec["counter1"] = 0
            spec["counter2"] = 0

            end_points = spec["end_points"]
            spec["first_point"] = (end_points[0], end_points[1])
            spec["second_point"] = (end_points[2], end_points[3])

            if spec["side1"] == "Negative":
                spec["enter_side1"] = "counter1"
                spec["exit_side1"] = "counter2"
            else:
                spec["enter_side1"] = "counter2"
                spec["exit_side1"] = "counter1"

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
    # Convert every new identified bounding box to an iten and returns the list of
    # new items.  Those items will not necessarily be added to the Setting.  The
    # identified bounding boxes could already be on the Setting or should not yet
    # be added to it.
    # ---------------------------------------------------------------------------------

    def _bboxes2items(self, bboxes, confidences, class_ids):

        new_inputs = []
        
        for (i, bbox) in enumerate(bboxes):
            item = Item(bbox[0], bbox[1], bbox[2], bbox[3], class_ids[i],
                        confidences[i])
            new_inputs.append(item)

        return new_inputs

    # ---------------------------------------------------------------------------------
    # When an item is created it initializes all the counting lines in the setting.
    # From an OO perspective, the counting lines should be created in the 'Setting'
    # class, but for every item we need to have all the counting lines and its
    # position in relation to the counting lines.  Is is easier to have the data
    # stored in the Item proper.  But need to be clearly documented.
    # ---------------------------------------------------------------------------------

    def _init_item_counters(self, item):
        item.lines = {}

        for key, value in self.cfg.data['counting_lines'].items():
            item.lines[key] = {}
            item.lines[key]['top_line_position'] = None
            item.lines[key]['bottom_line_position'] = None
            item.lines[key]['position'] = None
            item.lines[key]['counted_frame'] = self.cfg.frame_number
            item.lines[key]['split'] = False
            item.lines[key]['counted'] = False
            item.lines[key]['top_point'] = (item.startX, item.startY)
            item.lines[key]['bottom_point'] = (item.endX, item.endY)

    # ---------------------------------------------------------------------------------
    # Count the objects crossing the 'counting lines' given in the configuration
    # file.
    # ---------------------------------------------------------------------------------

    def _count(self):

        # for every counting line
        for key, spec in self.cfg.data["counting_lines"].items():
            # for every item, see if it has crossed the counting line
            for item in self.items:
                obj_line = object.lines[key]
                end_points = spec["end_points"]
                try: 
                    new_top = Geom.point_position(
                        end_points[0], end_points[1], end_points[2], end_points[3],
                        object.startX, object.startY)
                except OverflowError:
                    logging.info(Util.br_time() +
                        ", count: new_top overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                        end_points[0], end_points[1],
                        end_points[2], end_points[3],
                        object.startX, object.startY, self.cfg.analyser_id)
                    new_top = obj_line['top_line_position']

                try:
                    new_bottom = Geom.point_position(
                        end_points[0], end_points[1], end_points[2], end_points[3],
                        object.endX, object.endY)
                except OverflowError:
                    logging.info(Util.br_time() +
                        ", count: new_bottom overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                        end_points[0], end_points[1],
                        end_points[2], end_points[3],
                        object.startX, object.startY, self.cfg.analyser_id)
                    new_bottom = obj_line["bottom_line_position"]

                if (spec['count_splits'] == 'True' and obj_line['split'] == True):
                    if (object.dirY == 'South' and
                        obj_line['top_line_position'] != new_top):
                        self.top_crossed(object, obj_line, spec, new_top)
                        obj_line['top_line_position'] = new_top
                        obj_line["bottom_line_position"] = new_bottom

                # did the bottom line cross the count line...
                if (obj_line["bottom_line_position"] != new_bottom):
                    # Has the object just entered the scene? 
                    if (obj_line['bottom_line_position'] == None):
                        # counting line splits the new identified object
                        if new_top != new_bottom:
                            # print("object: " + str(object.object_id) + " is split")
                            obj_line['split'] = True
                    else:
                        self.bottom_crossed(object, obj_line, spec, new_bottom)
                        
                    obj_line['top_line_position'] = new_top
                    obj_line["bottom_line_position"] = new_bottom

                obj_line['top_point'] = (object.startX, object.startY)
                obj_line['bottom_point'] = (object.endX, object.endY)
                                
            
