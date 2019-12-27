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
import cv2
import mmap
import numpy as np
import math
import dlib

from thespian.actors import ActorSystem

from object_flow.ipc.doer import Doer

from object_flow.flow.item import Item
from object_flow.flow.setting import Setting

#==========================================================================================
# A Tracker tracks itens in a frame
#==========================================================================================

class Tracker(Doer):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def __init__(self, id):
        # every Doer should call super().__init__() if it has an __init__ method
        super().__init__()

        # this tracker id
        self.id = id
        
        # A Tracker can track itens from different videos, so it has a list of all
        # videos and for each video, the list of itens it is tracking
        self.video_analysers = {}
        
    # ----------------------------------------------------------------------------------
    # Returns the id of this partial tracker
    # ----------------------------------------------------------------------------------

    def get_id(self):
        return self.id

    # ----------------------------------------------------------------------------------
    # Starts a dlib tracker to track the object given by its bounding box. Receives
    # the 'video_analyser' as parameter and will keep track of all objects by camera.
    # _start_tracker is started when the process receives a 'Start' message.
    # TODO: uses explicitly the dlib tracker.  Should configure so that another tracker
    # could be used.
    # ----------------------------------------------------------------------------------

    def start_tracking(self, analyser_id, frame, object_id, startX, startY, endX, endY):

        # gets the correct list of video analyser objects
        va_objs = self.video_analysers.get(analyser_id, {})
        
        dlib_tracker = dlib.correlation_tracker()
        rect = dlib.rectangle(startX, startY, endX, endY)
        dlib_tracker.start_track(frame, rect)
        
        # add this dlib tracker to the list of tracked objects by this tracker
        va_objs[object_id] = dlib_tracker







    
    # ----------------------------------------------------------------------------------
    # This method is the main loop for the partial tracking process.
    # ----------------------------------------------------------------------------------

    def tracker(self):
        logging.info("Starting tracker %d", self.id)

        for data in iter(self.comm_q_in.get, None):
            logging.debug("%s, %d, %s, got message: %s",
                          Util.br_time(), self.id, os.getpid(), data[0])
            
            # data[0] contains the type of message received. Can be 'Start', 'Update',
            # 'Remove'
            if data[0] == 'Start':
                # data[1] = analyser_id
                # data[2] = frame
                # data[3] = object_id
                # data[4] = startX
                # data[5] = startY
                # data[6] = endX
                # data[7] = endY
                # print("process " + str(os.getpid()) + " with id: " + str(self.id) +
                #       " got " + str(data[0]))
                logging.debug(Util.br_time() +
                    ", Tracker %d: tracking object %d for camera %s", self.id, data[3], data[1])
                self.__start_tracker(data[1], data[2], data[3], data[4], data[5], data[6], data[7])
            elif data[0] == 'Update':
                # data[1] = analyser_id
                # data[2] = frame
                logging.debug(Util.br_time() +
                    ", Tracker %d: updating for camera %s", self.id, data[1])
                self.__update_trackers(data[1], data[2])
            elif data[0] == "Remove":
                # data[1] = analyser_id
                # data[2] = object_id
                logging.debug(Util.br_time() +
                    ", Tracker %d: removing object %d for camera %s", self.id, data[2], data[1])
                self.__remove_tracked(data[1], data[2])
            elif data[0] == 'End':
                logging.info("%s, %d, %s, %s",
                             Util.br_time(), self.id, os.getpid(), 
                             "Shutting down tracker. Received 'End' message")
                break
            elif data[0] == "Info":
                for key, value in self.video_analysers.items():
                    if key != "info":
                        total_items = len(self.video_analysers[key])
                        logging.info("%s, %d, %s, tracking %d objects for analyser %s",
                                     Util.br_time(), self.id, os.getpid(), total_items, key)
                p = psutil.Process()
                with p.oneshot():
                    logging.info("%s, %d, %s, cpu_times: %s, cpu_percent: %s, memory: %s",
                                 Util.br_time(), self.id, os.getpid(), p.cpu_times(),
                                 p.cpu_percent(), p.memory_info())
            elif data[0] == 'Test':
                logging.info("%s, %d, %s, Hello from server %d",
                             Util.br_time(), self.id, os.getpid(), self.id)
            else:
                logging.debug(Util.br_time() +
                              ", Unknown message type: " + data[0])
                

        
