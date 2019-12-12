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

from thespian.actors import Actor
from thespian.actors import ActorExitRequest
from thespian.actors import ChildActorExited
from thespian.actors import PoisonMessage
from thespian.actors import ActorSystemConventionUpdate
from thespian.actors import WakeupMessage

from object_flow.util.util import Util
from object_flow.ipc.memo import Memo
from object_flow.ipc.hr import HR

#==========================================================================================
#
#==========================================================================================

class Doer(HR, Actor):

    # ----------------------------------------------------------------------------------
    # Hires a new 'doer' for the 'group'
    # ----------------------------------------------------------------------------------
    
    def hire(self, name, klass, *args, group = 'default',
             target_actor_requirements = None, global_name = None, **kwargs):
        
        self.check_group(group)

        doer = self.createActor(klass, target_actor_requirements, global_name)
        self._doers[group][name] = doer

        if 'initialize' in dir(klass):
            logging.info("==========+++++++++++++++++++++")
            logging.info("I have an initialize method")
            method = getattr(klass, 'initialize')
            self.phone(doer, 'initialize', *args, **kwargs, callback = 'init_done') 
        
        return doer
        
    # ----------------------------------------------------------------------------------
    # init_done is a noop method that is called after an actor ends its 'initialize'
    # method.  If an actor has an 'initialize' method this methos is called
    # synchronously after the 'hire' method is called.
    # ----------------------------------------------------------------------------------

    def init_done(self, response):
        # logging.info("init_done called with response %s", response)
        pass
    
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
        self.send(address, memo)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def phone(self, address, method, *args, callback = None, reply_to = None, **kwargs):
        if callback == None:
            logging.debug("%s, %d, error: asking requires a callback function",
                          Util.br_time(), os.getpid())
        else:
            memo = Memo(method, *args, memo_type = 'ask', callback = callback,
                        reply_to = reply_to, **kwargs)
            self.send(address, memo)
            
    # ----------------------------------------------------------------------------------
    # Tells 'what' to 'whom' and does not wait for any reply.  'whom' is the doer and
    # group is the doer's group.  'who' is a String containing the name of the
    # doer that should receive the message
    # ----------------------------------------------------------------------------------

    def tell(self, whom, method, *args, group = 'default', **kwargs):
        memo = Memo(method, *args, memo_type = 'tell', **kwargs)
        for address in self._addresses(whom, group):
            self.send(address, memo)

    # ----------------------------------------------------------------------------------
    # 'who' is a String containing the name of the doer that should receive the message
    # ----------------------------------------------------------------------------------

    def ask(self, whom, method, *args, group = 'default', callback = None,
            reply_to = None, **kwargs):
        if callback == None:
            logging.debug("%s, %d, error: asking requires a callback function",
                          Util.br_time(), os.getpid())
        else:
            # create a Memo from the give parameters
            memo = Memo(method, *args, memo_type = 'ask', callback = callback,
                        reply_to = reply_to, **kwargs)
            for address in self._addresses(whom, group):
                self.send(address, memo)

    # ----------------------------------------------------------------------------------
    # Sends a message to the sender of the last message, unless the message has a
    # reply to field. In this case, the response should be sent to the reply_to
    # address
    # ----------------------------------------------------------------------------------

    def _response(self, return_value):
        
        callback = self.last_message._callback
        
        if isinstance(return_value, tuple):
            memo = Memo(callback, *return_value, memo_type = 'reply')
        else:
            memo = Memo(callback, return_value, memo_type = 'reply')
                        
        if self.last_message._reply_to != None:
            recipient = self.last_message._reply_to
        else:
            recipient = self.last_message_sender
            
        self.send(recipient, memo)
        
    # ----------------------------------------------------------------------------------
    # Sends a message to the sender of the last message
    # ----------------------------------------------------------------------------------

    def send_back(self, method, *args, **kwargs):
        memo = Memo(method, *args, memo_type = 'tell', **kwargs)
        self.send(self.last_message_sender, memo)
                
    # ----------------------------------------------------------------------------------
    # Abstract method 
    # ----------------------------------------------------------------------------------

    def actor_exit_request(self, message, sender):
        logging.info("%s, %d, got actor_exit_request",
                     Util.br_time(), os.getpid())
        pass
    
    # ----------------------------------------------------------------------------------
    # Abstract method for dealing with a child that has exited
    # ----------------------------------------------------------------------------------

    def child_actor_exited(self, message, sender):
        pass
    
    # ----------------------------------------------------------------------------------
    # Abstract method to deal with poison messages
    # ----------------------------------------------------------------------------------

    def poison_message(self, message, sender):
        pass
    
    # ----------------------------------------------------------------------------------
    # Abstract method to deal with actor system convention update messages
    # ----------------------------------------------------------------------------------

    def actor_system_convention_update(self, message, sender):
        pass
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def wakeup(self):
        pass
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    def receiveMessage(self, message, sender):
        # logging.debug("%s, %d, got message: %s",
        #               Util.br_time(), os.getpid(), message)

        self.last_message = message
        self.last_message_sender = sender

        if isinstance(message, Memo):
            # logging.debug("%s, %d, got a Memo: %s",
            #               Util.br_time(), os.getpid(), message._method)
            
            method = getattr(self, message._method)
            ret = method(*message._args, **message._kwargs)

            # if it's as 'ask' message, then we should reply to it
            if message._memo_type == 'ask':
                self._response(ret)

        elif isinstance(message, ActorExitRequest):
            self.actor_exit_request(message, sender)
        elif isinstance(message, ChildActorExited):
            self.child_actor_exited(message, sender)
        elif isinstance(message, PoisonMessage):
            self.poison_message(message, sender)
        elif isinstance(message, ActorSystemConventionUpdate):
            self.actor_system_convetion_update(message, sender)
        elif isinstance(message, WakeupMessage):
            self.wakeup()

        # delete the 'last_message' since it has already been processed
        self.last_message = None
        self.last_message_sender = None
            
