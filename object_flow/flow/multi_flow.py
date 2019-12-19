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
from object_flow.flow.flow_manager import FlowManager
from object_flow.nn.yolov3_tf2.yolotf2 import YoloTf2

#==========================================================================================
# VideoSupervisor supervises as many FlowManagers as there are video cameras.
# VideoSupervisor will create FlowManagers for every camera and supervise all external
# requests.  For instance, when the user whats to see a video, it should ask the
# VideoSupervisor
#==========================================================================================

class MultiFlow(Doer):
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def initialize(self):
        logging.basicConfig(filename='myapp.log', level=logging.INFO)
        
        self._yolo = self.hire('YoloNet', YoloTf2, group = 'DeepLearners')
        self.main()
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def hired(self, hiree_name, hiree_group, hiree_address):
        if hiree_group == 'DeepLearners':
            logging.info("%s, %s, %s, %s", Util.br_time(), "all", os.getpid(), 
                         "Yolo neural net ready to roll!")
        if hiree_group == 'flow_manager':
            pass
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def flow_manager_initialized(self, video_name):
        self.start_playback(video_name)
    
    # ----------------------------------------------------------------------------------
    # Adds a new camera to be processes.  It creates a camera manager and let's it do
    # its work
    # ----------------------------------------------------------------------------------

    def add_camera(self, video_name, path):
        # create th e camera manager and initialize it with video_name and
        # the Yolo neural net
        manager = self.hire(video_name, FlowManager, video_name, path, self._yolo,
                            group = 'flow_manager')
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def start_playback(self, video_name):
        self.tell(video_name, 'start_playback', group = 'flow_manager')
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def stop_playback(self, video_name):
        self.tell(video_name, 'stop_playback', group = 'flow_manager')
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def main(self):
        self.add_camera('Vivo', 'resources/videos/Vivo.avi')
        self.add_camera('Shopping3', 'resources/videos/shopping3.avi')
