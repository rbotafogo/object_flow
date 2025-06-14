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
import pprint

from thespian.actors import ActorSystem

from object_flow.util.util import Util
from object_flow.ipc.memo import Memo
from object_flow.ipc.hr import HR

#==========================================================================================
#
#==========================================================================================

class Board(HR):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def __init__(self, proc = 'tcp', logcfg = None):
        super().__init__()
        
        if proc == 'tcp':
            logging.info("Starting Actor System with mutiplrocTCPBase")
            self._system = ActorSystem('multiprocTCPBase', logDefs=logcfg)
        elif proc == 'udp':
            logging.info("Starting Actor System with mutiplrocUDPBase")
            self._system = ActorSystem('multiprocUDPBase', logDefs=logcfg)
        else:
            logging.info("Wrong Actor System type")
            raise('Invalid system type definition')

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def init_done(self, response):
        pass
    
    # ----------------------------------------------------------------------------------
    # Hires a new 'doer' for the 'group'
    # ----------------------------------------------------------------------------------
    
    def hire(self, name, klass, *args, group = 'default',
             target_actor_requirements = None, global_name = None, source_hash = None,
             **kwargs):

        logging.info("Hiring new doer of class %s with name %s in group %s", klass, name,
                     group)
        
        self.check_group(group)

        doer = self._system.createActor(
            klass, target_actor_requirements, global_name, source_hash)
        self._doers[group][name] = doer

        # self.phone(doer, '__set_id__', *args, _name_ = name, _group_ = group,
        #            **kwargs, callback = '__hired__')

        if '__initialize__' in dir(klass):
            method = getattr(klass, '__initialize__')
            self.phone(doer, '__initialize__', *args, **kwargs, callback = 'init_done')
            
        return doer

    # ----------------------------------------------------------------------------------
    # Sends a message to the given address.  Creates a Memo from the *args and
    # **kwargs with memo_type = 'tell'
    # @param address [actorAddress]: an actor address
    # @param method [String] method name
    # @param [*args]
    # @param [**kwargs]
    # ----------------------------------------------------------------------------------

    def post(self, address, method, *args, **kwargs):
        memo = Memo(method, *args, memo_type = 'tell', **kwargs)
        self._system.tell(address, memo)
        
    # ----------------------------------------------------------------------------------
    # Tells 'what' to 'who' and does not way for any reply.  'who' is the doer and
    # group is the doer's group.  'who' is a String containing the name of the
    # doer that should receive the message
    # ----------------------------------------------------------------------------------

    def tell(self, whom, method, *args, group = 'default', **kwargs):
        memo = Memo(method, *args, memo_type = 'tell', **kwargs)
        
        for address in self._addresses(whom, group):
            self._system.tell(address, memo)

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def phone(self, address, method, *args, timeout = None, callback = None,
              reply_to = None, **kwargs):
        if callback == None:
            logging.info("%s, %d, error: asking requires a callback function",
                         Util.br_time(), os.getpid())
        else:
            memo = Memo(method, *args, memo_type = 'ask', callback = callback,
                        reply_to = reply_to, **kwargs)
            self._system.ask(address, memo, timeout)
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def ask(self, whom, method, *args, group = 'default', timeout = None,
            callback = None, reply_to = None, **kwargs):
        if callback == None:
            logging.info("%s, %d, error: asking requires a callback function",
                         Util.br_time(), os.getpid())
        else:
            # create a Memo from the give parameters
            memo = Memo(method, *args, memo_type = 'ask', callback = callback,
                        reply_to = reply_to, **kwargs)
            for address in self._addresses(whom, group):
                self._system.ask(address, memo, timeout)

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def all_doers_address(self):
        for group in self._doers:
            for doer in self._doers[group].items():
                yield doer[1]
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def shutdown(self):
        for doer_address in self.all_doers_address():
            self.post(doer_address, 'actor_exit_request', 'exit', doer_address)
        
        self._system.shutdown()
