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
    import time
    import threading
    import logging

    from object_flow.ipc.board import Board
    from object_flow.flow.multi_flow import MultiFlow

    logging.basicConfig(filename='myapp.log', level=logging.INFO)
    
    board = Board()
    board.hire('MultiFlow', MultiFlow)

    time.sleep(120)
    board.shutdown()

# import ctypes
# kernel32 = ctypes.windll.kernel32

## This sets the priority of the process to realtime--the same priority as the mouse pointer.
# kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 31)
## This creates a timer. This only needs to be done once.
# timer = kernel32.CreateWaitableTimerA(ctypes.c_void_p(), True, ctypes.c_void_p())
## The kernel measures in 100 nanosecond intervals, so we must multiply .25 by 10000
# delay = ctypes.c_longlong(.25 * 10000)
# kernel32.SetWaitableTimer(timer, ctypes.byref(delay), 0, ctypes.c_void_p(), ctypes.c_void_p(), False)
# kernel32.WaitForSingleObject(timer, 0xffffffff)
