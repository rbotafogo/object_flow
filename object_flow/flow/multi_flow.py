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
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019, 2020
##########################################################################################

import os
import time
import logging

from object_flow.ipc.doer import Doer
from object_flow.util.util import Util
from object_flow.util.config import Config
from object_flow.nn.yolov3_tf2.yolotf2 import YoloTf2
from object_flow.flow.flow_manager import FlowManager
from object_flow.flow.tracker import Tracker

from object_flow.util.mmap_bboxes import MmapBboxes

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

    def __init__(self):
        super().__init__()
        self.nn_ready = False
        # TODO: the name should be set by the hiring doer.  In this case the
        # hiring doer is the Board and all the Board communication (ipc) needs
        # to be improved
        self.name = 'MultiFlow'
        self.group = 'default'
        
        # if of the flow_manager
        self._next_flow_id = 0

        # Create the memory maped file for communicating bounding boxes
        self._mmap_bbox = MmapBboxes()
        self._mmap_bbox.create()

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def __initialize__(self, system_cfg):
        self.system_cfg = system_cfg
        
        # load confidence and threshold from the specific neural net algo
        confidence = system_cfg.data['neural_net']['confidence']
        threshold = system_cfg.data['neural_net']['threshold']

        # start the yolo object detection process
        self._yolo = self.hire('YoloNet', YoloTf2, confidence, threshold,
                               group = 'DeepLearners')

        # read from conf file how many trackers we want and create the trackers
        self.ntrackers = system_cfg.data['system_info']['num_trackers']
        self.add_trackers(self.ntrackers)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __hired__(self, hiree_name, hiree_group, hiree_address):
        if hiree_group == 'trackers':
            logging.info("New tracker %s hired", hiree_name)
            # _main should only run after all the trackers and the neural net have
            # being instantiated. We call _main multiple times but check if
            # ntrackers == 0 (all trackers hired) and that nn_ready is True
            self._main()
        if hiree_group == 'DeepLearners':
            logging.info("Yolo neural net ready to roll")
            self.nn_ready = True
            # _main should only run after all the trackers and the neural net have
            # being instantiated. We call _main multiple times but check if
            # ntrackers == 0 (all trackers hired) and that nn_ready is True
            self._main()
        if hiree_group == 'flow_manager':
            logging.info("New flow_manager %s hired", hiree_name)
            # self._test_tracker_communication()
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # SERVICES

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

    def add_tracker(self):
        self.ntrackers -= 1
        self.hire('Tracker_' + str(self.ntrackers), Tracker, id = self.ntrackers,
                  tracker_type = self.system_cfg.data['system_info']['tracker_type'],
                  group = 'trackers')
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def add_trackers(self, num):
        for i in range(num):
            self.add_tracker()
    
    # ----------------------------------------------------------------------------------
    # Adds a new camera to be processes.  It creates a camera manager and let's it do
    # its work
    # ----------------------------------------------------------------------------------

    def add_camera(self, cfg):
        # create the flow manager and initialize it with video_name and
        # the Yolo neural net
        cfg.start_time = self.system_cfg.data['system_info']['start_time']
        cfg.minutes = self.system_cfg.data['system_info']['minutes']
        manager = self.hire(
            cfg.video_name, FlowManager, cfg, self._doers['trackers'],
            self._yolo, self._next_flow_id,
            group = 'flow_manager')
        self._next_flow_id += 1
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # SYNCHRONIZATION METHODS

    # ----------------------------------------------------------------------------------
    # This method is called by a flow_manager just after it has been initialized to
    # let multi_flow know that it can use this flow_manager
    # ----------------------------------------------------------------------------------

    def flow_manager_initialized(self, video_name):
        self.start_playback(video_name)
        # pass
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS
    
    # ----------------------------------------------------------------------------------
    # Main control flow of multi_flow: for every camera in the 'video_cameras'
    # configuration section of the system configuration file, read the specific
    # configuration file for the camera and start the camera.
    # ----------------------------------------------------------------------------------

    def _main(self):
        if not(self.nn_ready == True and self.ntrackers == 0):
            return
        
        videos = self.system_cfg.data['video_cameras']
        
        for j, video in enumerate(videos):
            logging.info("Reading configuration file for video %s", video)
            logging.info("Analytics will be output to: %s",
                         self.system_cfg.data['system_info']['analytics_output_dir'])
            config_file = (
                os.path.dirname(self.system_cfg.data['system_info']['config_dir']) + "/" +
                videos[video])
            
            cfg = MultiFlow._read_configuration_file(
                config_file, self.system_cfg.data['system_info'])
            cfg.video_name = os.path.splitext(os.path.basename(videos[video]))[0]
            cfg.delta_csv_update = self.system_cfg.data['system_info']['delta']

            self.add_camera(cfg)
            
    # ----------------------------------------------------------------------------------
    # Reads the configuration file for the video specific video. It first reads the
    # default configuration file and then reads the specific file. 
    # ----------------------------------------------------------------------------------
    
    def _read_configuration_file(config_file, system_info):

        # Create configuration object and loads the defaults configuration file
        cfg = Config("config/defaults.json")
        cfg.output_dir = system_info['analytics_output_dir']
        
        # Load specific configuration file
        if config_file:
            # set the name of the configuration file in the config object
            cfg.config_file = config_file
            cfg.update(config_file)

        # add to the configuration object the file name (without extension) of
        # the configuration file
        cfg.file_name = os.path.splitext(os.path.basename(config_file))[0]
        cfg.file_path = cfg.output_dir + "/" + Util.brus_datetime() + "_" + cfg.file_name

        cfg.system_info = system_info

        # check to see if 'record' parameter is set in the config file
        if 'record' in cfg.data["video_processor"]:
            cfg.data["video_processor"]["record"] = (
                cfg.data["video_processor"]["record"] == 'True')
        # if 'record' not set in config, then check its value on the command
        # line parameters.  The config file takes precedence over the
        # command line parameters.
        else:
            cfg.data["video_processor"]["record"] = system_info['record']

        # 'record' parameter set to true. Check the start time and end time of
        # recording
        if cfg.data["video_processor"]["record"]:
            cfg.record_file_name = os.path.basename(cfg.data["io"]["record"])
            
            cfg.data['video_processor']['start_time'] = "00:01"
            cfg.data['video_processor']['end_time'] = "23:59"
            # check if record should happen on a given weekday
            if 'record_weekday' in cfg.data['video_processor']:
                record_weekday = cfg.data['video_processor']['record_weekday']
                today = Util.isoweekday()
                delta_day = (record_weekday - today) % 7
            
            # check if there is an 'record_time' parameter in the config file
            if 'record_time' in cfg.data["video_processor"]:
                record_time = cfg.data["video_processor"]["record_time"].split('-')
                # fix the start_time
                time = record_time[0].strip().split(':')
                cfg.data['video_processor']['start_time'] = (
                    Util.set_tzaware_time(int(time[0], 10), int(time[1], 10),
                                          delta_day = delta_day))
                # fix the end_time
                time = record_time[1].strip().split(':')
                cfg.data['video_processor']['end_time'] = (
                    Util.set_tzaware_time(int(time[0], 10), int(time[1], 10),
                                          delta_day = delta_day))
                logging.info("recording for %s will start on: %s",
                             cfg.file_name,
                             cfg.data['video_processor']['start_time'])
                logging.info("recording for %s will end on: %s", 
                             cfg.file_name,
                             cfg.data['video_processor']['end_time'])

        # same type of check done for 'edit' parameter
        if 'edit' in cfg.data["video_processor"]:
            cfg.data["video_processor"]["edit"] = (
                cfg.data["video_processor"]["edit"] == 'True')
        else:
            cfg.data["video_processor"]["edit"] = system_info['edit']

        # Correct the 'show_id' configuration parameter
        cfg.data["video_processor"]["show_id"] = (
            cfg.data["video_processor"]["show_id"] == 'True')        
        # Correct the 'show_input_bbox' configuration parameter
        cfg.data["video_processor"]["show_input_bbox"] = (
            cfg.data["video_processor"]["show_input_bbox"] == 'True')
        # Correct the 'show_tracking_bbox' configuration parameter
        cfg.data["video_processor"]["show_tracking_bbox"] = (
            cfg.data["video_processor"]["show_tracking_bbox"] == "True")
        
        return cfg

    # ----------------------------------------------------------------------------------
    # broadcast to all trackers a 'say_hello' message
    # ----------------------------------------------------------------------------------

    def _test_tracker_communication(self):
        for tracker in self._doers['trackers']:
            self.tell(tracker, 'say_hello', 1, 2, 3, a = 4, b = 5, group='trackers')
        
