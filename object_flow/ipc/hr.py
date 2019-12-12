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

from object_flow.util.util import Util

#==========================================================================================
#
#==========================================================================================

class HR:
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def __init__(self):
        self._doers = {}
        self._doers['default'] = {}
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def check_group(self, group):
        if not group in self._doers:
            logging.info("%s, %d, creating new group: %s",
                         Util.br_time(), os.getpid(), group)
            self._doers[group] = {}

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    def hrreport(self):
        logging.info("%s, %d, doers reporting to me: %s",
                     Util.br_time(), os.getpid(), self._doers)

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def hrreport2(self):
        self.reply(self._doers)

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def _addresses(self, whom, group = 'default'):
        # send the memo to everyone
        if whom == 'all':
            for group in self._doers:
                for doer in self._doers[group]:
                    yield doer
        # send the memo to group
        elif whom == None:
            for doer in self._doers[group].items():
                yield doer[1]
        elif whom in self._doers[group]:
            yield self._doers[group][whom]
        else:
            logging.info("%s, %d, doer: %s is not in the group: %s",
                          Util.br_time(), os.getpid(), whom, group)
        
