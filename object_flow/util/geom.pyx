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
import scipy
import scipy.spatial

class Geom:

    # ----------------------------------------------------------------------------------
    # Returns False in one side of the line and true on the other.  The line is
    # given by the two bounding points: first_point and second_point
    # ----------------------------------------------------------------------------------

    def point_position(int fp0, int fp1, int sp0, int sp1,
                       int x, int y):
        return ((sp0 - fp0) * (y - fp1) - (sp1 - fp1) * (x - fp0)) > 0

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def ccw(int p10, int p11, int p20, int p21, int p30,
            int p31):
	# return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
        return ((p31 - p11) * (p20 - p10) > (p21 - p11) * (p30 - p10))

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def intersect(int p10, int p11, int p20, int p21,
                  int p30, int p31, int p40, int p41):
	# return Geom.ccw(A,C,D) != Geom.ccw(B,C,D) and Geom.ccw(A,B,C) != Geom.ccw(A,B,D)
        return (Geom.ccw(p10, p11, p30, p31, p40, p41) !=
                Geom.ccw(p20, p21, p30, p31, p40, p41) and
                Geom.ccw(p10, p11, p20, p21, p30, p31) !=
                Geom.ccw(p10, p11, p20, p21, p40, p41))
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # def in_range(p1, p2, p3):
    #     return ((p1[0] > p2[0] and p1[0] < p3[0]) and
    #             (p1[1] > p2[1] and p1[1] < p3[1]))

    # ---------------------------------------------------------------------------------
    # Calculates the intersection over union of this bounding box with another
    # given bounding box
    # ---------------------------------------------------------------------------------

    def iou(int box_startX, int box_startY, int box_endX,
            int box_endY, int obox_startX, int obox_startY,
            int obox_endX, int obox_endY):
        # determine the (x, y)-coordinates of the intersection rectangle
        x_start = max(box_startX, obox_startX)
        y_start = max(box_startY, obox_startY)
        x_end = min(box_endX, obox_endX)
        y_end = min(box_endY, obox_endY)

        # compute the area of intersection rectangle
        inter_area = max(0, x_end - x_start + 1) * max(0, y_end - y_start + 1)

        # compute the area of the given box. The area of the current object
        # is always kept updated
        box_area = ((box_endX - box_startX + 1) * (box_endY - box_startY + 1))

        # compute the area of the given box. The area of the current object
        # is always kept updated
        obox_area = ((obox_endX - obox_startX + 1) * (obox_endY - obox_startY + 1))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box_area + obox_area - inter_area)

        # return the intersection over union value
        return iou
	
    # ---------------------------------------------------------------------------------
    # matches the input objects with the trackable objects by verifying the ones that
    # have higher intersection over union score
    # ---------------------------------------------------------------------------------

    def iou_match(set1, set2, match_value):
        iou_array = np.zeros((len(set1), len(set2)), dtype="float64")

        # calculate the iou metric for all tracked objects x input objects
	# creates a matrix of iou distances
        for (i, to) in enumerate(set1):
            for (j, io) in enumerate(set2):
                iou_array[i, j] = Geom.iou(to.startX, to.startY, to.endX, to.endY,
                                           io.startX, io.startY, io.endX, io.endY)

        # TODO: should this be calculated automatically somehow
        match = np.where(iou_array > match_value)
        rows = np.unique(match[0])
        cols = []
        match_rows_cols = []

        for row in rows:
            col = iou_array[row, :].argmax()
            if col in cols:
                continue
            cols.append(col)
            match_rows_cols.append((row, col))

        allrows = np.arange(len(set1))
        unused_rows = np.setdiff1d(allrows, rows)
	
        unused_cols = set(range(0, iou_array.shape[1])).difference(cols)
        return unused_rows, unused_cols, match_rows_cols

    # ---------------------------------------------------------------------------------
    # matches the input objects with trackable objects by verifying the ones for which
    # the centroid is closer
    # ---------------------------------------------------------------------------------
    
    def centroid_match(set1, set2, max_dist):
        c_array = np.zeros((len(set1), len(set2)), dtype="float64")

        # calculate the iou metric for all tracked objects x input objects
	# creates a matrix of iou distances
        for (i, to) in enumerate(set1):
            for (j, io) in enumerate(set2):
                c_array[i, j] = scipy.spatial.distance.euclidean(to.centroid, io.centroid)

        # TODO: should this be calculated automatically somehow
        match = np.where(c_array < max_dist)
        rows = np.unique(match[0])
        cols = []
        match_rows_cols = []

        for row in rows:
            col = c_array[row, :].argmin()
            if col in cols:
                continue
            cols.append(col)
            match_rows_cols.append((row, col))

        allrows = np.arange(len(set1))
        unused_rows = np.setdiff1d(allrows, rows)
	
        unused_cols = set(range(0, c_array.shape[1])).difference(cols)
        return unused_rows, unused_cols, match_rows_cols
    



    
    # ----------------------------------------------------------------------------------
    # Returns False in one side of the line and true on the other.  The line is
    # given by the two bounding points: first_point and second_point
    # ----------------------------------------------------------------------------------

    def point_position2(first_point, second_point, x, y):
        position = ((second_point[0] - first_point[0]) *
                    (y - first_point[1]) -
                    (second_point[1] - first_point[1]) *
                    (x - first_point[0])) > 0
        return position

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def ccw2(p1, p2, p3):
	# return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)
        return ((p3[1] - p1[1]) * (p2[0] - p1[0]) >
                (p2[1] - p1[1]) * (p3[0] - p1[0]))

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def intersect2(p1, p2, p3, p4):
	# return Geom.ccw(A,C,D) != Geom.ccw(B,C,D) and Geom.ccw(A,B,C) != Geom.ccw(A,B,D)
        return (Geom.ccw(p1, p3, p4) != Geom.ccw(p2, p3, p4) and
                Geom.ccw(p1, p2, p3) != Geom.ccw(p1, p2, p4))
    
    # ---------------------------------------------------------------------------------
    # Calculates the intersection over union of this bounding box with another
    # given bounding box
    # ---------------------------------------------------------------------------------

    def iou2(box, other_box):
        # determine the (x, y)-coordinates of the intersection rectangle
        x_start = max(box.startX, other_box.startX)
        y_start = max(box.startY, other_box.startY)
        x_end = min(box.endX, other_box.endX)
        y_end = min(box.endY, other_box.endY)

        # compute the area of intersection rectangle
        inter_area = max(0, x_end - x_start + 1) * max(0, y_end - y_start + 1)

        # compute the area of the given box. The area of the current object
        # is always kept updated
        other_box_area = ((other_box.endX - other_box.startX + 1) *
                          (other_box.endY - other_box.startY + 1))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = inter_area / float(box.area + other_box_area - inter_area)

        # return the intersection over union value
        return iou
	
