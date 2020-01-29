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

# needed to open the mmap file
import logging
from datetime import timedelta

from object_flow.ipc.doer import Doer

# CHECK_PERIOD = timedelta(milliseconds=30)

#==========================================================================================
#
#==========================================================================================

class DrumBeat(Doer):

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()

        # list of listeners interested to get a message everytime a new frame is
        # loaded
        self._listeners = {}

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def __initialize__(self, video_name, speed):
        self.video_name = video_name
        self.check_period = speed
        
        self.wakeup()

    # ----------------------------------------------------------------------------------
    # Adds a new listener to this drum beater.
    # ----------------------------------------------------------------------------------

    def add_listener(self, name, address):
        logging.info("%s: adding listener to drum beater with name %s",
                     self.video_name, name)
        self._listeners[name] = address
    
    # ----------------------------------------------------------------------------------
    # Removes the listener
    # ----------------------------------------------------------------------------------

    def remove_listener(self, name):
        logging.info("%s: listener %s removed from drum beater",
                     self.video_name, name)
        del self._listeners[name]
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def wakeup(self):
        self._notify_listeners()
        self.wakeupAfter(self.check_period)
    
    # ----------------------------------------------------------------------------------
    # increments the check_period by 'mili' milliseconds.  
    # ----------------------------------------------------------------------------------

    def inc_check_period(self, milli):
        inc = timedelta(milliseconds=milli)
        self.check_period += inc
        logging.info("%s: drum beat check period is now %s milliseconds",
                     self.video_name, str(self.check_period))
        
    # ----------------------------------------------------------------------------------
    # decrements the check_period by 'mili' milliseconds.  
    # ----------------------------------------------------------------------------------

    def dec_check_period(self, milli):
        inc = timedelta(milliseconds=milli)
        self.check_period -= inc
        logging.info("%s: drum beat check period is now %s milliseconds",
                     self.video_name, str(self.check_period))
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _notify_listeners(self):
        
        # notify every listener that we have a new frame and give it the
        # buffer size
        for name, listener in self._listeners.items():
            # listener: doer's address
            # when sending the base image, send also all the items, so that they
            # can be used by other methods
            self.post(listener, 'capture_next_frame')
            
