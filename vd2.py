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
    import os
    import sys
    import datetime
    from pytz import timezone
    import logging
    import argparse

    from _version import __version__

    from object_flow.ipc.board import Board
    from object_flow.flow.multi_flow import MultiFlow
    from object_flow.util.util import Util
    from object_flow.util.config import Config

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('--version', action='version',
                    version='%(prog)s {version}'.format(version=__version__))
    
    # specific configuration file
    ap.add_argument(
        "-c", "--config",
        help="path to the configuration file for all videos",
        required=True)
    
    # arguments needed for video processing
    ap.add_argument(
        "-e", "--edit", default=False,
	help="allow editing of the video to add counting lines")
    
    ap.add_argument(
        "-v", "--video",
        help="pass a single video for processing and ignore the videos in the configuration file")
    
    ap.add_argument(
        "-o", "--output", default=False,
        help="set to True if the processed video should be output to a file. The file name is defined in the configuration file for this video")
    
    ap.add_argument(
        "-m", "--minutes", type = int, default=10,
        help="Number of minutes between every analytics output on the csv file")
    
    ap.add_argument(
        
        "-w", "--with_min", default=True,
        help="Should the csv file have minutes in it")
    
    ap.add_argument(
        "-s", "--start_time",
        help="Time of the first csv data generation. Should be of the form HH:MM")
    
    # arguments needed for the neural net
    ap.add_argument(
        "-p", "--process",
        help="The neural net processing engine to use, either 'opencv' or 'tf2'")
    
    args = vars(ap.parse_args())

    # Create configuration object and loads the defaults configuration file
    logging.info("%s, %s, %s, %s", Util.br_time(), args['config'], os.getpid(), 
                 "Reading configuration file")

    # cfg = Config(args["config"])
    cfg = Config("config/system_defaults.json")
    cfg.update(args['config'])

    cfg.data['system_info']['config_dir'] = args['config']
    cfg.data['system_info']['edit'] = (args['edit'] == 'True')
    cfg.data['system_info']['output'] = (args['output'] == 'True')
    cfg.data['system_info']['minutes'] = args['minutes']

    now = Util.br_time_raw()
    
    if args['start_time'] != None:
        start_time = datetime.datetime.strptime(args['start_time'], '%H:%M')
        hour = start_time.hour
        minute = start_time.minute
        start_time = Util.set_tzaware_time(hour, minute)
    else:
        start_time = Util.round_dt(now, 10)

    logging.info(start_time)
    
    if (start_time > now):
        cfg.data['system_info']['delta'] = datetime.timedelta(
            seconds=((start_time - now).total_seconds()))
    else:
        cfg.data['system_info']['delta'] = datetime.timedelta(
            minutes=cfg.data['system_info']['minutes'])

    logging.info("%s, %s, %s, timedelta for first csv update is: %s",
                 Util.br_time(), "all", os.getpid(), 
                 str(cfg.data['system_info']['delta']))
        
    cfg.data['system_info']['with_min'] = (
        args['with_min'] == 'True' or args['with_min'] == True)

    # check if the -v/--video parameter was given
    if args['video']:
        cfg.data['video_cameras'] = {}
        cfg.data['video_cameras'][0] = args['video']

    # Configure the Neural Net. By default use the 'tf2' configuration
    if args["process"]:
        if (args['process'] != 'tf2' and args['process'] != 'opencv' and
            args['process'] != 'tnets'):
            raise Exception(
                "Process flag should either be 'tf2', 'opencv' or 'tnets'")
        cfg.data["neural_net"]["process"] = args["process"]
    else:
        cfg.data['neural_net']['process'] = 'tf2'


    class actorLogFilter(logging.Filter):
        def filter(self, logrecord):
            return 'actorAddress' in logrecord.__dict__
        
    class notActorLogFilter(logging.Filter):
        def filter(self, logrecord):
            return 'actorAddress' not in logrecord.__dict__
        
    logcfg = { 'version': 1,
               'formatters': {
                   'normal': {'format': '%(levelname)-8s %(message)s'},
                   'actor': {'format': '%(levelname)-8s %(actorAddress)s => %(message)s'}},
               'filters': { 'isActorLog': { '()': actorLogFilter},
                            'notActorLog': { '()': notActorLogFilter}},
               'handlers': { 'h1': {'class': 'logging.FileHandler',
                                    'filename': 'example.log',
                                    'formatter': 'normal',
                                    'filters': ['notActorLog'],
                                    'level': logging.INFO},
                             'h2': {'class': 'logging.FileHandler',
                                    'filename': 'example.log',
                                    'formatter': 'actor',
                                    'filters': ['isActorLog'],
                                    'level': logging.INFO},},
               'loggers' : { '': {'handlers': ['h1', 'h2'], 'level': logging.DEBUG}}
    }
    
    logcfg = { 'version': 1,
               'formatters': {
                   'normal': {
                       # 'format': "%(asctime)s;%(levelname)s;%(message)s"}},
                       'format': "%(levelname)-6s;%(asctime)s;%(filename)s;%(funcName)s;%(lineno)d;%(process)d;%(message)s", 'datefmt': '%Y-%m-%d;%H:%M:%S'}
               },
               'handlers': {
                   'h': {'class': 'logging.FileHandler',
                         'filename': 'log/flow.log',
                         'formatter': 'normal',
                         'level': logging.INFO
                   },
                   'console': {'class': 'logging.StreamHandler',
                               'formatter': 'normal',
                               'stream': 'ext://sys.stdout',
                               'level': logging.INFO
                   },
               },
               'loggers' : {
                   '': {'handlers': ['h', 'console'],
                        'level': logging.DEBUG}}
    }
    
    board = Board(logcfg=logcfg)
    board.hire('MultiFlow', MultiFlow, cfg)

    time.sleep(30)
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
