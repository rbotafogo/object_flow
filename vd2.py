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

    # from object_flow.neural_nets.yolov3_tf2.yolotf2 import YoloTf2
    from object_flow.nn.yolov3_tf2.yolotf2 import YoloTf2
    from object_flow.ipc.board import Board
    from object_flow.flow.multi_flow import MultiFlow

    def vivo():
        board.tell('MultiFlow', 'add_camera', 'Vivo', 'resources/videos/Vivo.avi')
        # give time for the add_camera to start
        time.sleep(1 )
        board.tell('MultiFlow', 'start_playback', 'Vivo')
        time.sleep(5)
        board.tell('MultiFlow', 'stop_playback', 'Vivo')
        time.sleep(5)
        board.tell('MultiFlow', 'start_playback', 'Vivo')

    def shopping3():
        board.tell('MultiFlow', 'add_camera', 'Shopping3',
                   'resources/videos/shopping3.avi')
        board.tell('MultiFlow', 'start_playback', 'Shopping3')

        
    board = Board()
    board.hire('MultiFlow', MultiFlow)

    time.sleep(10)
    
    thread1 = threading.Thread(target = vivo, args = ())
    thread2 = threading.Thread(target = shopping3, args = ())

    thread1.start()
    thread2.start()
    
    time.sleep(60)
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
