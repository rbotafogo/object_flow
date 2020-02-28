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

import time
import logging


class Stopwatch:

    # dictionary of time measures for metrics reporting
    _measures = {}
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    @classmethod
    def start(cls, measure):
        if not measure in cls._measures:
            cls._measures[measure] = {}
            cls._measures[measure]['total'] = 0
            cls._measures[measure]['num_events'] = 0

        cls._measures[measure]['start'] = time.perf_counter()
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    @classmethod
    def stop(cls, measure):
        now = time.perf_counter()
        elapsed = now - cls._measures[measure]['start']
        
        cls._measures[measure]['num_events'] += 1
        cls._measures[measure]['total'] += elapsed 
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    @classmethod
    def clear(cls, measure):
        cls._measures[measure]['total'] = 0
        cls._measures[measure]['num_events'] = 0
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    @classmethod
    def report(cls, name, total_frames, main_measure = None, num_frames = 100):
        
        if total_frames % num_frames == 0:
            total = 0
            for measure in cls._measures:
                num_events = cls._measures[measure]['num_events']
                logging.info("%s: average time to " + measure + " for the last " +
                             str(num_frames) + " frames is %f", name,
                             cls._measures[measure]['total'] / num_frames)
                if num_events != num_frames:
                    logging.info("%s: average time to " + measure + " for the last " +
                                 str(num_events) + " events is %f", name,
                                 cls._measures[measure]['total'] / num_events)
                if measure != main_measure:
                    total += cls._measures[measure]['total'] / num_frames
                elif main_measure != None:
                    process_total = cls._measures[measure]['total'] / num_frames

                cls.clear(measure)
                
            logging.info("%s: total time spent %f", name, total)
            
            if main_measure != None:
                logging.info("%s: time unaccounted for %s", name,
                             process_total - total)
                
            logging.info('=================================')
    

