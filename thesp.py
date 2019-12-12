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

if __name__ == '__main__':
    import logging
    import sys
    import time

    # basic elements for the extended Thespian system
    from object_flow.ipc.board import Board

    # example of a doer
    from object_flow.ipc.examples import CEO
    from object_flow.ipc.examples import Eng
    
    board = Board()
    
    board.hire('john', CEO)
    # board.hire('mary', CEO)
    # board.hire('paul', CEO, 'engineering')
    # board.hire('anne', CEO, 'engineering')

    board.tell('john', 'build_team')
    
    # board.tell('john', 'say_hello', 'board')
    # board.tell('mary', 'say_hello', 'Happy board')
    # board.tell(None, 'say_hello', 'Big board')
    # board.tell(None, 'say_hello', 'engineer board', group = 'engineering')
    # board.tell('anne', 'say_hello', '=====', group = 'engineering')
    
    time.sleep(10)
    board.shutdown()
    
