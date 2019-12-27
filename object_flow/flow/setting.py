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

from object_flow.flow.item import Item

# =========================================================================================
# Class Setting is responsible for managing all the items in the camera view
# =========================================================================================

class Setting:

    def __init__(self):
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
    # Add new detected elements to the setting if they are not already tracked and
    # returns the added elements as Items
    # ---------------------------------------------------------------------------------

    def add_detections(self, bboxes, confidences, class_ids):

        # Checks if the item should be added/removed from the Setting.  Items should
        # only be in the Setting if they are inside the entry lines.
        # bboxes = self.validate_entry(bboxes)
        new_inputs = self.bboxes2items(bboxes, class_ids, confidences)

        # TODO: lots of things....!!!
        self.items = new_inputs
