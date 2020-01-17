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
            for key in self.cfg.data['counting_lines']:
                item.init_lines(key, self.cfg.frame_number)

    # ---------------------------------------------------------------------------------
    # does all required updates on the received bounding_box after this bounding_box
    # has changed position from the tracking algorithm
    # ---------------------------------------------------------------------------------

    def update(self, bounding_box):
        self._check_exit(bounding_box)
        self._count()
    
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
    # checks if the item has crossed an entry_line and is exiting the setting
    # ---------------------------------------------------------------------------------

    def _check_exit(self, bounding_box):

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

    def _should_count(self, obj_line):
        
        if (not obj_line['counted'] or
            (self.cfg.frame_number > obj_line['counted_frame'] + 30)):
            # once an object crosses the counting line it is no longer considered
            # a split object
            obj_line['split'] = False
            obj_line['counted'] = True
            obj_line['counted_frame'] = self.cfg.frame_number
            return True
        return False
        
    # ---------------------------------------------------------------------------------
    # Top line of the bounding box has crossed the counting line. This is called
    # for split objects going 'South': we will count them when the top line crosses
    # the counting line, since the bottom line will not cross the counting line
    # ---------------------------------------------------------------------------------

    def _top_crossed(self, item, item_line, spec, new_top, track_id = None):
        top_point = item_line['top_point']
        first_point = spec['first_point']
        second_point = spec['second_point']

        try:
            if (not Geom.intersect(
                    top_point[0], top_point[1], item.startX,
                    item.startY, first_point[0], first_point[1],
                    second_point[0], second_point[1])):
                if (item.item_id == track_id):
                    logging.info("item %d did not cross over the line", item.item_id)
                return
        
        except OverflowError:
            logging.info(
                "top_crossed: overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                top_point[0], top_point[1], item.startX,
                item.startY, first_point[0], first_point[1],
                second_point[0], second_point[1], self.video_name)
        
        if (item.item_id == track_id):
            logging.debug("should we count this item? %d", item.item_id)
            
        if (self._should_count(item_line)):
            if new_top:
                logging.debug("counting +1 for item %d", item.item_id)
                spec[spec["exit_side1"]] += 1
        
    # ---------------------------------------------------------------------------------
    # Bottom line of the bounding box has crossed the counting line
    # ---------------------------------------------------------------------------------

    def _bottom_crossed(self, item, item_line, spec, new_bottom):
        bottom_point = item_line['bottom_point']
        first_point = spec['first_point']
        second_point = spec['second_point']

        try:
            if (not Geom.intersect(bottom_point[0], bottom_point[1], item.endX,
                                   item.endY, first_point[0], first_point[1],
                                   second_point[0], second_point[1])):
                return
            
        except OverflowError:
            logging.info(
                "bottom_crossed: overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                bottom_point[0], bottom_point[1], item.endX,
                item.endY, first_point[0], first_point[1],
                second_point[0], second_point[1], self.video_name)

                    
        if (self._should_count(item_line)):
            if not new_bottom:
                spec[spec["enter_side1"]] += 1
            else:
                spec[spec["exit_side1"]] += 1

    # ---------------------------------------------------------------------------------
    # Count the items crossing the 'counting lines' given in the configuration
    # file.
    # ---------------------------------------------------------------------------------

    def _count(self):

        track_id = None
        
        # for every counting line
        for key, spec in self.cfg.data["counting_lines"].items():
            # for every item, see if it has crossed the counting line
            for item_id, item in self.items.items():
                item_line = item.lines[key]
                end_points = spec["end_points"]
                try: 
                    new_top = Geom.point_position(
                        end_points[0], end_points[1], end_points[2], end_points[3],
                        item.startX, item.startY)
                    if (item_id == track_id):
                        logging.debug("item %d new_top is %s in relation to line %s", item_id, new_top, key)
                except OverflowError:
                    logging.warning(
                        "count: new_top overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                        end_points[0], end_points[1],
                        end_points[2], end_points[3],
                        item.startX, item.startY, self.video_name)
                    new_top = item_line['top_line_position']

                try:
                    new_bottom = Geom.point_position(
                        end_points[0], end_points[1], end_points[2], end_points[3],
                        item.endX, item.endY)
                    if (item_id == track_id):
                        logging.debug("item %d new_bottom is %s in relation to line", item_id, new_bottom, key)
                except OverflowError:
                    logging.warning(
                        "count: new_bottom overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                        end_points[0], end_points[1],
                        end_points[2], end_points[3],
                        item.startX, item.startY, self.video_name)
                    new_bottom = item_line["bottom_line_position"]

                if (spec['count_splits'] == 'True' and item_line['split'] == True):
                    if (item_id == track_id):
                        logging.debug("item %d direction is %s", item_id, item.dirY)
                    if (item.dirY == 'South' and
                        item_line['top_line_position'] != new_top):
                        if (item_id == track_id):
                            logging.info("item %d top has crossed the counting line %s", item_id, key)
                        self._top_crossed(item, item_line, spec, new_top, track_id)
                        item_line['top_line_position'] = new_top
                        item_line["bottom_line_position"] = new_bottom

                # did the bottom line cross the count line...
                if (item_line["bottom_line_position"] != new_bottom):
                    # Has the item just entered the scene? 
                    if (item_line['bottom_line_position'] == None):
                        # counting line splits the new identified item
                        if new_top != new_bottom:
                            # print("item: " + str(item.item_id) + " is split")
                            item_line['split'] = True
                    else:
                        if (item_id == track_id):
                            logging.info("item %d bottom has crossed the counting line %s", item_id, key)
                        self._bottom_crossed(item, item_line, spec, new_bottom)
                        
                    item_line['top_line_position'] = new_top
                    item_line["bottom_line_position"] = new_bottom

                item_line['top_point'] = (item.startX, item.startY)
                item_line['bottom_point'] = (item.endX, item.endY)
                                
            
