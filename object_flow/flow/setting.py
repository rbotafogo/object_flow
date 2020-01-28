 # -*- coding: utf-8 -*-

##########################################################################################
# @author Rodrigo Botafogo
#
# Copyright (C) 2019 Rodrigo Botafogo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019
##########################################################################################

import itertools
import logging

from object_flow.flow.item import Item
from object_flow.util.geom import Geom
from object_flow.flow.csv import CSV

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

        CSV.initialize(cfg)
        
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
    # After tracking is done, for each tracked item, update_item is called so that
    # it's bounding_box is adjusted to the tracker's information
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
    # Check if two trackable objects have an iou grater than a given value. If they
    # have, return the index of the trackable object that overlaps the given object
    # ---------------------------------------------------------------------------------

    def _has_overlap(self, item, min_index):

        keys = list(self.items.keys())
        
        for i in range(min_index, len(keys)):
            ti = self.items[keys[i]]
            try: 
                iou = Geom.iou(item.startX, item.startY, item.endX, item.endY,
                               ti.startX, ti.startY, ti.endX, ti.endY,)
            except OverflowError:
                logging.info("%s: has_overlap overflow: (%d, %d, %d, %d)-(%d, %d, %d, %d)",
                             self.video_name, 
                             item.startX, item.startY, item.endX, item.endY,
                             ti.startX, ti.startY, ti.endX, ti.endY)
                iou = 0
            
            if (iou > self.cfg.data['trackable_objects']['drop_overlap'] and
                  item.direction == ti.direction):
                return keys[i]
            
        return -1
    
    # ---------------------------------------------------------------------------------
    # Checks all tracked objects and drop those that have similar bounding boxes.
    # Tracked objects and input objects can be the same, but be seen as different
    # since matching by iou has lots of false positive.
    # ---------------------------------------------------------------------------------

    def find_overlap(self):

        overlapped = []
        
        begin = 0
        keys = list(self.items.keys())
        while begin < len(keys):
            # If two tracked objects overlap then mark the last one to be removed, if they
            # are both going the same direction. If they are going different directions,
            # they are probably not the same object.  Method __has_overlap does this
            # check
            overlap = self._has_overlap(self.items[keys[begin]], begin + 1)
            if (overlap > 0):
                overlapped.append(overlap)
                
            begin += 1

        return overlapped
    
    # ---------------------------------------------------------------------------------
    # does all required updates on the received bounding_box after this bounding_box
    # has changed position from the tracking algorithm
    # ---------------------------------------------------------------------------------

    def update(self, bounding_box):
        # should we remove overlapping bounding boxes? If we do so, then occluded
        # items will be removed; on the other hand, objects detected twice will be
        # eliminated
        # self._drop_overlap()

        # count items that have crossed any counting lines
        self._count()
        CSV.csv_schedule(self.cfg)
    
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
    # Convert every new identified bounding box to an iten and returns the list of
    # new items.  Those items will not necessarily be added to the Setting.  The
    # identified bounding boxes could already be on the Setting or should not yet
    # be added to it.
    # ---------------------------------------------------------------------------------

    def _bboxes2items(self, bboxes, confidences, class_ids):

        new_inputs = []
        
        for (i, bbox) in enumerate(bboxes):
            item = Item(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                        class_ids[i], confidences[i])
            new_inputs.append(item)

        return new_inputs

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def _overflow_warning(self, where, end_points, p1, p2):
        message = where + " overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s"
        logging.warning(
            message, end_points[0], end_points[1], end_points[2], end_points[3],
            p1, p2, self.cfg.video_name)
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def _position_item_line(self, end_points, pX, pY):
        try: 
            position = Geom.point_position(
                end_points[0], end_points[1], end_points[2], end_points[3],
                pX, pY)
        except OverflowError:
            self._overflow_warning("check_exit: new_top", end_points,
                                   pX, pY)
            position = False
            
        return position

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def _positions_box_line(self, box):
        for key, spec in self.cfg.data["entry_lines"].items():
            end_points = spec["end_points"]
            top = self._position_item_line(end_points, box[0], box[1])
            bottom = self._position_item_line(end_points, box[2], box[3])

            yield (spec, top, bottom)
    
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

            for spec, top, bottom in self._positions_box_line(box):
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
    # checks if the item has crossed an entry_line and is exiting the setting
    # ---------------------------------------------------------------------------------

    def _check_exit(self, bounding_box):

        for spec, new_top, new_bottom in self._positions_box_line(bounding_box):
            # side1 indicates the valid region for entry, that is, the area
            # in which if the object´s bounding box is completely inside
            # then it should be seen.
            # a split object in relation to an entry line should not be allowed
            if (new_top != new_bottom):
                return True

        return False
                    
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

    def _top_crossed(self, item, item_line, spec, new_top):
        top_point = item_line['top_point']
        first_point = spec['first_point']
        second_point = spec['second_point']

        try:
            if (not Geom.intersect(top_point[0], top_point[1], item.startX,
                                   item.startY, first_point[0], first_point[1],
                                   second_point[0], second_point[1])):
                return
        
        except OverflowError:
            logging.warning(
                "top_crossed: overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                top_point[0], top_point[1], item.startX,
                item.startY, first_point[0], first_point[1],
                second_point[0], second_point[1], self.cfg.video_name)
            
        if (self._should_count(item_line)):
            if new_top:
                spec[spec["exit_side1"]] += 1
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def _has_top_crossed(self, item, key, spec, new_top):
        item_line = item.lines[key]

        if (spec['count_splits'] == 'True' and item_line['split'] == True and
            item.dirY == 'South' and item_line['top_line_position'] != new_top):
            self._top_crossed(item, item_line, spec, new_top)
        
            # debugging information
            if item.item_id == self.track_item:
                logging.info("frame_number %d: item %d bottom line has crossed line %s",
                             self.cfg.frame_number, item.item_id, key)
                if item_line['split'] == True:
                    logging.info("item %d is a split item in relation to line %s",
                                 item.item_id, key)
            # end debugging information

    
    # ---------------------------------------------------------------------------------
    # Bottom line of the bounding box has crossed the counting line
    # ---------------------------------------------------------------------------------

    def _bottom_crossed(self, item, item_line, spec, new_bottom):
        bottom_point = item_line['bottom_point']
        first_point = spec['first_point']
        second_point = spec['second_point']

        logging.debug("checking bottom_crossed for camera %s", self.cfg.video_name)

        if item.item_id == self.track_item:
            logging.info("frame_number %d: item %d counting bottom crossed",
                         self.cfg.frame_number, item.item_id)
                        
        try:
            if (not Geom.intersect(bottom_point[0], bottom_point[1], item.endX,
                                   item.endY, first_point[0], first_point[1],
                                   second_point[0], second_point[1])):
                
                if item.item_id == self.track_item:
                    logging.info("frame_number %d: item %d returning without counting",
                                 self.cfg.frame_number, item.item_id)
                return
            
        except OverflowError:
            logging.warning(
                "bottom_crossed: overflow error: (%d, %d, %d, %d)-(%d, %d) for camera %s",
                bottom_point[0], bottom_point[1], item.endX,
                item.endY, first_point[0], first_point[1],
                second_point[0], second_point[1], self.cfg.video_name)

                    
        if (self._should_count(item_line)):
            if not new_bottom:
                spec[spec["enter_side1"]] += 1
            else:
                spec[spec["exit_side1"]] += 1
        
    # ---------------------------------------------------------------------------------
    # Checks if the bottom line has crossed the counting line. If this is a new
    # item, then check to see if this item was split by the counting line. As split
    # item should be counted, if it is going down the image, when the top line
    # crosses the counting line
    # ---------------------------------------------------------------------------------

    def _has_bottom_crossed(self, item, key, spec, new_bottom): 
        item_line = item.lines[key]
                        
        # did the bottom line cross the count line. Bottom has crossed the line
        # and the direction was set
        if (item_line["bottom_line_position"] != new_bottom and
            ((item.dirY != None) or (item.dirX != None))):
            self._bottom_crossed(item, item_line, spec, new_bottom)
            
            # debugging information
            if item.item_id == self.track_item:
                logging.info("frame_number %d: item %d bottom line has crossed line %s",
                             self.cfg.frame_number, item.item_id, key)
                if item_line['split'] == True:
                    logging.info("item %d is a split item in relation to line %s",
                                 item.item_id, key)
            # end debugging information
            
    # ---------------------------------------------------------------------------------
    # returns the position of the top and bottom lines of the bounding box in relation
    # to a given line. (startX, startY) is checked as top line and (endX, endY) is
    # considered as the bottom line
    # ---------------------------------------------------------------------------------

    def _find_positions(self, item, spec):
        
        end_points = spec["end_points"]
        # find the top position in relation to the given line
        new_top = self._position_item_line(end_points, item.startX, item.startY)
        # find the bottom position in relation to the given line
        new_bottom = self._position_item_line(end_points, item.endX, item.endY)
        
        return(new_top, new_bottom)
    
    # ---------------------------------------------------------------------------------
    # Count the items crossing the 'counting lines' given in the configuration
    # file.
    # ---------------------------------------------------------------------------------

    def _count(self):

        # this value should be set by a service in flow_manager, through a UI
        self.track_item = None

        # for every counting line
        for key, spec in self.cfg.data["counting_lines"].items():
            # for every item, see if it has crossed the counting line
            logging.debug("counting items is respect to line %s", key)
            for item_id, item in self.items.items():
                logging.debug("checking item %d", item_id)
                
                new_top, new_bottom = self._find_positions(item, spec)

                if (item.lines[key]['bottom_line_position'] != None):
                    # bottom line has crossed the counting line?
                    self._has_bottom_crossed(item, key, spec, new_bottom)
                    
                    # top line has crossed the counting line?
                    self._has_top_crossed(item, key, spec, new_top)
                else:
                    if new_top != new_bottom:
                        item.lines[key]['split'] = True

                item.lines[key]['top_line_position'] = new_top
                item.lines[key]["bottom_line_position"] = new_bottom
                
