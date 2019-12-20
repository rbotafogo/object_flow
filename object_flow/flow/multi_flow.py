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
from object_flow.util.config import Config
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
    
    def initialize(self, system_cfg):
        self.system_cfg = system_cfg
        
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
        logging.info("===================================================")
        logging.info("===================================================")
        logging.info("===================================================")
        logging.info("===================================================")
        
        videos = self.system_cfg.data['video_cameras']
        
        for j, video in enumerate(videos):
            logging.info("%s, %s, %s, %s", Util.br_time(), videos[video], os.getpid(), 
                         "Reading configuration file")
            logging.info("%s, %s, %s, %s %s", Util.br_time(), videos[video], os.getpid(), 
                         "Analytics will be output to: ",
                         self.system_cfg.data['system_info']['analytics_output_dir'])
            config_file = (
                os.path.dirname(self.system_cfg.data['system_info']['config_dir']) + "/" +
                videos[video])
            
            cfg = MultiFlow._read_configuration_file(
                config_file, self.system_cfg.data['system_info'])
            cfg.analyser_id = os.path.splitext(os.path.basename(videos[video]))[0]
            cfg.delta_csv_update = self.system_cfg.data['system_info']['delta']

            logging.info("===================================================")
            logging.info("===================================================")
            logging.info("===================================================")
            logging.info(cfg.analyser_id)
            logging.info(cfg.data['io']['input'])
            
            self.add_camera(cfg.analyser_id, cfg.data['io']['input'])
            
        # self.add_camera('Shopping3', 'resources/videos/shopping3.avi')
    # ----------------------------------------------------------------------------------
    #
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
        # if 'record' no set in config, then check its value on the command
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
                logging.info("%s, %s, %s, recording will start on: %s", Util.br_time(),
                             cfg.file_name, os.getpid(),
                             cfg.data['video_processor']['start_time'])
                logging.info("%s, %s, %s, recording will end on: %s", Util.br_time(),
                             cfg.file_name, os.getpid(),
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

        return cfg
            
