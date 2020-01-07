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
from datetime import timedelta

from object_flow.util.util import Util

from object_flow.ipc.doer import Doer

#==========================================================================================
#
#==========================================================================================

class Eng(Doer):
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __initialize__(self, boss, salary):
        logging.info("initializing Eng with boss %s and salary %f", boss, salary)
        
        self.boss = boss
        self.salary = salary
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def say_hello(self, name):
        logging.debug("%s, %d, hello %s from your CEO",
                      Util.br_time(), os.getpid(), name)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def sum(self, val1, val2):
        logging.debug("%s, %d, Your engineer working for you",
                      Util.br_time(), os.getpid())
        return (val1 + val2)

#==========================================================================================
#
#==========================================================================================

class CEO(Doer):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def say_hello(self, name):
        logging.debug("%s, %d, hello %s from your CEO",
                      Util.br_time(), os.getpid(), name)

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def sum_result(self, total):
        logging.info("%s, %d, total of sum is %d",
                     Util.br_time(), os.getpid(), total)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def build_team(self):
        eng1 = self.hire('eng1', Eng, 'Mary', 20000, group = 'engineering')
        self.hire('eng2', Eng, 'john', salary = 1500, group = 'engineering')
        # self.hrreport()

        # ask my engineers to sum... let's see if they are any good!
        self.ask(None, 'sum', 2, 3, group = 'engineering', callback = 'sum_result')

        # ask by giving an address
        self.phone(eng1, 'sum', 2, 3, callback = 'sum_result')
