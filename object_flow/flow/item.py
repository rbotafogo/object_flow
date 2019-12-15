# -*- coding: utf-8 -*-

##########################################################################################
# @author Rodrigo Botafogo
#
# Copyright (C) 2019 Rodrigo Botafogo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019
##########################################################################################

from collections import deque

# =========================================================================================
# Defines a bounding box by its starting and ending points
# A bounding box can have an 'id' which can be anything
# =========================================================================================

class Item:

    def __init__(self, startX, startY, endX, endY,
                 class_id = None, confidence = None):

        # Coordinates of the bounding box
        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

        # the bounding box will receive an object id for the duration
        # of the streaming video. The tracking algorithm is responsible
        # for registering the bounding box and giving it an object id
        self.object_id = None

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
        
        # which partial tracker is tracking this object
        self.partial_tracker_id = None

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
        self.update_centroid()

    # ---------------------------------------------------------------------------------
    # Updates the centroid of this tracked object
    # ---------------------------------------------------------------------------------

    def update_centroid(self):

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

    def update_bbox(self, startX, startY, endX, endY):

        self.startX = startX
        self.startY = startY
        self.endX = endX
        self.endY = endY

        # update the centroid
        self.update_centroid()

