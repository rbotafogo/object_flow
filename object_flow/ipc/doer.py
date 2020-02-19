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
from thespian.initmsgs import *

from object_flow.ipc.memo import Memo
from object_flow.ipc.hr import HR

#==========================================================================================
#
#==========================================================================================

class Doer(Actor):

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()
        
        self._doers = {}
        self._doers['default'] = {}
        
    # ----------------------------------------------------------------------------------
    # After a Doer has initialized, going throuhg '_set_id_' and '__initialize__' the
    # hiring Doer receives the 'hired' message with the name, group and address of the
    # hiree
    # ----------------------------------------------------------------------------------

    def __hired__(self, hiree_name, hiree_group, hiree_address):
        pass
    
    # ----------------------------------------------------------------------------------
    # When a Doer is hired, it imediately gets the '_set_id_' message, with its name
    # and group (given by the hiring doer) and eventually some parameters to use for
    # calling the '__initialize__' method.  The '__initialize__' method can be added to
    # any Doer to do any necessary initialization that requires runtime information.
    # Initializations that do not require runtime information can be done in the
    # __init__ method normaly
    # ----------------------------------------------------------------------------------

    def __set_id__(self, *args, _name_ = None, _group_ = None, _parent_ = None, **kwargs):
        self.name = _name_
        self.group = _group_
        self.parent_address = _parent_

        if '__initialize__' in dir(self):
            method = getattr(self, '__initialize__')
            method(*args, **kwargs)
        
    # ----------------------------------------------------------------------------------
    # Hires a new 'doer' for the 'group'
    # ----------------------------------------------------------------------------------
    
    def hire(self, name, klass, *args, group = 'default',
             target_actor_requirements = None, global_name = None, **kwargs):
        
        self.check_group(group)
        
        doer = self.createActor(klass, target_actor_requirements, global_name)
        self._doers[group][name] = (doer, klass)

        self.phone(doer, '__set_id__', *args, **kwargs, _name_ = name, _group_ = group,
                   _parent_ = self.myAddress, callback = '__hired__', memo_type = 'hire')

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
        self.send(address, memo)
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def phone(self, address, method, *args, callback = None, reply_to = None,
              memo_type = 'ask', **kwargs):
        if callback == None:
            logging.debug("%s, %s: asking requires a callback function", self.name,
                          self.group)
        else:
            memo = Memo(method, *args, memo_type = memo_type, callback = callback,
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
            logging.debug("asking requires a callback function")
        else:
            # create a Memo from the give parameters
            memo = Memo(method, *args, memo_type = 'ask', callback = callback,
                        reply_to = reply_to, **kwargs)
            for address in self._addresses(whom, group):
                self.send(address, memo)

    # ----------------------------------------------------------------------------------
    # Sends a message to the sender of the last message
    # ----------------------------------------------------------------------------------

    def send_back(self, method, *args, **kwargs):
        memo = Memo(method, *args, memo_type = 'tell', **kwargs)
        self.send(self.last_message_sender, memo)
                
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def all_doers_address(self):
        for group in self._doers:
            for doer in self._doers[group].items():
                yield doer[0], group, doer[1][0]
            
    # ----------------------------------------------------------------------------------
    # send to all doers the 'actor_exit_request'. In principle this should not be
    # necessary, but in many cases Python processes keep running even after the
    # main Admin has shutdown
    # ----------------------------------------------------------------------------------

    def terminate(self):
        for doer_name, doer_group, doer_address  in self.all_doers_address():
            logging.info("%s-%s: sending actor_exit_request to: %s-%s",
                         self.name, self.group, doer_name, doer_group)
            self.send(doer_address, 'actor_exit_request')
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def actor_exit_request(self, message, sender):
        logging.info("%s-%s: got actor_exit_request", self.name, self.group)
        self.terminate()
    
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
        # logging.debug("got message: %s", message)

        self.last_message = message
        self.last_message_sender = sender

        if isinstance(message, Memo):
            # logging.debug("%s, %s: got a Memo: %s", self.name, self.group, message._method)
            # logging.debug("%s, %s: receiveMessage: calling method %s", self.name,
            #               self.group, message._method)
            method = getattr(self, message._method)
            ret = method(*message._args, **message._kwargs)

            # if it's as 'ask' message, then we should reply to it
            if message._memo_type == 'ask':
                # logging.debug("%s, %s, %s, %s", Util.br_time(), "all", os.getpid(), 
                #               "receiveMessage sending response")
                self._response(ret)
            elif message._memo_type == 'hire':
                # logging.debug("%s, %s, %s, %s", Util.br_time(), "all", os.getpid(), 
                #               "receiveMessage hiring done")
                self._response((self.name, self.group, self.myAddress))

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
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def check_group(self, group):
        if not group in self._doers:
            logging.info("%s-%s: creating new group: %s", self.name, self.group, group)
            self._doers[group] = {}

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    def hrreport(self):
        logging.info("%s-%s: doers reporting to me: %s", self.name, self.group,
                     self._doers)

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
                    yield doer[0]
        # send the memo to group
        elif whom == None:
            for doer in self._doers[group].items():
                yield doer[1][0]
        elif whom in self._doers[group]:
            yield self._doers[group][whom][0]
        else:
            logging.info("doer: %s is not in the group: %s", whom, group)

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
        
            
