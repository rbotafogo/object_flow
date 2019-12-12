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

import os
import time
import logging
# import mmap

from object_flow.util.util import Util
from object_flow.ipc.doer import Doer
from object_flow.decoder.video_manager import VideoManager
from object_flow.nn.yolov3_tf2.yolotf2 import YoloTf2

#==========================================================================================
# VideoSupervisor supervises as many VideoManagers as there are video cameras.
# VideoSupervisor will create VideoManagers for every camera and supervise all external
# requests.  For instance, when the user whats to see a video, it should ask the
# VideoSupervisor
#==========================================================================================

class VideoSupervisor(Doer):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def initialize(self):
        self._yolo = self.hire('YoloNet', YoloTf2)
    
    # ----------------------------------------------------------------------------------
    # Adds a new camera to be processes.  It creates a camera manager and let's it do
    # its work
    # ----------------------------------------------------------------------------------

    def add_camera(self, video_name, path):
        # create the camera manager and initialize it with video_name and
        # the Yolo neural net
        manager = self.hire(video_name, VideoManager, video_name, path, self._yolo)
        
        # post a message to the manager just created for it to start processing
        self.post(manager, 'run')
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def start_playback(self, video_name):
        self.tell(video_name, 'start_playback')
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def stop_playback(self, video_name):
        self.tell(video_name, 'stop_playback')
        
