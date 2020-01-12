# -*- coding: utf-8 -*-

##########################################################################################
# @author Rodrigo Botafogo
#
# Copyright (C) 2019 Rodrigo Botafogo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019
##########################################################################################

import numpy as np

from collections import deque

# =========================================================================================
# Defines a bounding box by its starting and ending points
# A bounding box can have an 'id' which can be anything
# =========================================================================================

class Item:

    # ---------------------------------------------------------------------------------
    # An item has a bounding box represented by its starting (x, y) point which lies
    # on the top left corner of the box and and ending point (x, y) which lies on the
    # right bottom corner of the box.  It also has a class id representing the type of
    # item, like person, car, etc. and the confidence
    # ---------------------------------------------------------------------------------

    def __init__(self, startX, startY, endX, endY, class_id = None, confidence = 1.0):

        # Coordinates of the bounding box
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

        # the bounding box will receive an object id for the duration
        # of the streaming video. The tracking algorithm is responsible
        # for registering the bounding box and giving it an object id
        self.item_id = None

        # class_id and confidence for this class id.  Those values are
        # given by the detection algorithm
        self.class_id = class_id
        self.confidence = confidence
        
        # Frame where the bounding box first appeared and when it was
        # last seen
        self.first_frame = None
        self.last_frame = None
        self.last_update = 0

        # indicates that this item has disappeared from the
        # video stream
        self.disappeared = False

        # dlib tracker to track this object
        self.tracker = None
        
        # direction to which the object is moving
        self.dirX = None
        self.dirY = None
        
        # create a list to store the last 32 centroids for
        # this Bounding box
        self.centroids = deque(maxlen=32)

        # how much this item has moved
        (self.dX, self.dY) = (0, 0)

        # string containing the direction of the movement
        self.direction = ""

        # update the centroid
        self._update_centroid()

    # ---------------------------------------------------------------------------------
    # Updates this item according to the information given by the tracker
    # ---------------------------------------------------------------------------------

    def tracker_update(self, frame_number, confidence, startX, startY, endX, endY):
        
        # update the dlib tracker
        self.confidence = confidence

        # updating the bbox also updates the centroid information used
        # bellow
        self._update_bbox(startX, startY, endX, endY)

        (self.dirX, self.dirY) = ("", "")

        # If the centroid moved far enough for the last 10 updates
        if len(self.centroids) > 10:
            self.dX = self.centroids[-10][0] - self.centroids[0][0]
            self.dY = self.centroids[-10][1] - self.centroids[0][1]

            if np.abs(self.dX) > 20:
                self.dirX = "East" if np.sign(self.dX) == 1 else "West"
            if np.abs(self.dY) > 20:
                self.dirY = "North" if np.sign(self.dY) == 1 else "South"

            # handle when both directions are non-empty
            if self.dirX != "" and self.dirY != "":
                self.direction = "{}-{}".format(self.dirY, self.dirX)

            # otherwise, only one direction is non-empty
            else:
                self.direction = self.dirX if self.dirX != "" else self.dirY

            # if the difference is too small, then set the last_update
            if np.abs(self.dX) < 20 and np.abs(self.dY) < 20:
                if self.last_update == 0:
                    self.last_update = frame_number
            else:
                self.last_update = 0
                
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    # PRIVATE METHODS
    
    # ---------------------------------------------------------------------------------
    # Updates the centroid of this tracked object
    # ---------------------------------------------------------------------------------

    def _update_centroid(self):

        # when updating the centroid, also update the area of
        # the bounding box
        self.area = (self.endX - self.startX + 1) * \
            (self.endY - self.startY + 1)

        # calculate the bounding box centroid
        self.cX = int((self.startX + self.endX) / 2.0)
        self.cY = int((self.startY + self.endY) / 2.0)
        self.centroid = (self.cX, self.cY)

        # add the centroid to the list of previous centroids
        self.centroids.appendleft(self.centroid)

    # ---------------------------------------------------------------------------------
    # Updates this object bounding box location to the new location given
    # by bbox
    # ---------------------------------------------------------------------------------

    def _update_bbox(self, startX, startY, endX, endY):

        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

        # update the centroid
        self._update_centroid()

