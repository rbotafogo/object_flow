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

#==========================================================================================
#
#==========================================================================================

class Memo(object):

    def __init__(self, method, *args, memo_type, callback = None, reply_to = None,
                 **kwargs):
        self._method = method
        self._args = args
        self._memo_type = memo_type
        self._callback = callback
        self._reply_to = reply_to
        self._kwargs = kwargs

#==========================================================================================
#
#==========================================================================================
