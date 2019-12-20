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

import json
import logging    

from dictdiffer import diff, patch, swap, revert

from object_flow.util.util import Util

# =========================================================================================
#
# =========================================================================================

class Config:

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def __init__(self, default_config):

        # read the default configuration file
        self.data = self.read_config(default_config)
        self.analyser_id = None

        # base name (without extension) of the configuration file.  This name will
        # be used on the csv output file.
        self.file_name = None
        
        # Keep the frame and framenumber in a shared object (this one) that is
        # accessed by all the other classes that require information about the
        # frame and frame_number
        self.frame = None
        self.frame_encoding = None
        self.frame_number = None

        # width and height of the frame
        self.width = None
        self.height = None

        # initialize the video stream, pointer to output video file
        # self.output = self.data["video_processor"]["output"]

        # file to write our processed video
        self.writer = None

        # file to write the csv output data
        self.csv_file = None

        # are the counter lines initialized?
        self.counter_lines_initialized = False

        # last time the csv file was updated
        self.last_csv_update = None
        self.delta_csv_update = None

        # individual detected queue out, for receiving detection information from 
        # the detection algorithm (neural net - GPU)
        self.detect_q_out = None
        self.detect_q_in = None
        self.detect_q_out_index = None

    # ---------------------------------------------------------------------------------
    # Convert configuration string parameter to boolean
    # ---------------------------------------------------------------------------------

    def fix_configurations(self):

        self.data["video_processor"]["show_input_bbox"] = (
            self.data["video_processor"]["show_input_bbox"] == 'True')
        self.data["video_processor"]["show_tracking_bbox"] = (
            self.data["video_processor"]["show_tracking_bbox"] == "True")
        self.data["video_processor"]["show_id"] = (
            self.data["video_processor"]["show_id"] == "True")
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def read_config(self, file_name):
        with open(file_name) as json_data_file:
            return json.load(json_data_file)
    
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def update(self, file_name):
        new_config = self.read_config(file_name)
        result2 = []

        for change in diff(self.data, new_config):
            if change[0] == 'change' or change[0] == "add":
                result2.append(change)

        self.data = patch(result2, self.data)
        self.fix_configurations()
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def get(self, key):
        return self.data[key]

