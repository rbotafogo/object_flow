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

# needed to open the mmap file
import os
import mmap
import math
import tempfile
import time
import collections
from datetime import timedelta
import cv2
import logging
import numpy as np
import random

from object_flow.ipc.doer import Doer
from object_flow.util.display import Display
from object_flow.util.geom import Geom

from object_flow.decoder.video_decoder import VideoDecoder
from object_flow.flow.item import Item
from object_flow.flow.setting import Setting
from object_flow.util.mmap_frames import MmapFrames
from object_flow.util.mmap_bboxes import MmapBboxes

#==========================================================================================
# FlowManager manages the process of decoding, detection and tracking for one video
# camera.  Every FlowManager has a Setting, i.e., the camera view with all the added
# information, such as the counting lines, the identified items, the bounding boxes,
# etc.
#==========================================================================================

class FlowManager(Doer):
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def __init__(self):
        super().__init__()

        # total number of frames processed by this FlowManager
        self._total_frames = 0
        
        # id of the next item
        self.next_item_id = 0

        # number of frames that can be stored in the mmaped file
        self._buffer_max_size = 500

        # list of listeners interested to get a message everytime a new frame is
        # loaded
        self._listeners = {}

        self.playback = False
        self.playback_started = False

        self.frame_index = 0
        self.temp_file=None
        self._time_load = 0
        self._time_ckd = 0
        self._time_removal = 0
        self._time_tracking = 0
        self._time_detection = 0
        self._time_notif = 0
        self._time_findbboxes = 0
        self._time_detect2box = 0
        self._time_add_items = 0
        self._total_time = 0

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def __initialize__(self, cfg, trackers, yolo, video_id):
        
        self.cfg = cfg
        # trackers hired by multi_flow, available to all flow_managers
        self.trackers = trackers
        self._yolo = yolo
        self.video_id = video_id
        self.video_name = cfg.video_name
        self.path = cfg.data['io']['input']
        self._last_detection = -self.cfg.data['video_analyser']['skip_detection_frames']
        logging.info("%s: initializing flow_manager with %s", self.video_name,
                     self.path)

        if self.cfg.is_image == True:
            writer = None
            filenum = len([lists for lists in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, lists))])
            fileid = 1
            output_path=self.path+'/'+self.video_name+'.avi'
            while fileid <= filenum:
                filename = str(fileid).rjust(6, '0') + ".jpg"
                frame = cv2.imread(self.path + '/' + filename)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(output_path, fourcc, 30,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
                fileid += 1
            writer.release()
            self.path=output_path
        # hire a new video decoder named 'self.video_name'
        self.vd = self.hire(self.video_name, VideoDecoder, self.video_name,
                            self.path, buffer_max_size = self._buffer_max_size,
                            group = 'decoders')

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def __hired__(self, hiree_name, hiree_group, hiree_address):
        if hiree_group == 'display':
            logging.info("%s: display hired", self.video_name)
        if hiree_group == 'decoders':
            logging.info("%s: decoder created", self.video_name)
            self.phone(hiree_address, 'get_image_info', callback = 'initialize_mmap')

    # ----------------------------------------------------------------------------------
    # send to all doers the 'actor_exit_request'. In principle this should not be
    # necessary, but in many cases Python processes keep running even after the
    # main Admin has shutdown
    # ----------------------------------------------------------------------------------

    def terminate(self):
        for doer_address in self.all_doers_address():
            self.send(doer_address, 'actor_exit_request')
        self._mmap.close()
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def actor_exit_request(self, message, sender):
        logging.info("%s, %s: got actor_exit_request", self.name, self.group)
        self.terminate()
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # SERVICES
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def start_playback(self):

        display  = self.video_name + '_display'
        self._dp = self.hire(display, Display, self.video_name, self.cfg,
                             group = 'displayers')

        logging.info("%s: starting playback", self.video_name)
        
        # initialize the display
        self.phone(self._dp, 'initialize_mmap', self.width, self.height, self.depth,
                   callback = '_add_listener')
                
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # TODO: remove video_name as a parameter
    def destroy_window(self, video_name):
        display = self.video_name + '_display'
        self.tell(display, 'destroy_window', group = 'displayers')
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def stop_playback(self):
        
        if self.playback == False:
            return
        elif self.playback_started == False:
            self.post(self.myAddress, 'stop_playback')
            return

        logging.info("stopping playback for video %s", self.video_name)
        
        self.remove_listener(self.video_name)
        self.destroy_window(self.video_name)
        
        self.playback = False
        self.playback_started = False
        
    # ----------------------------------------------------------------------------------
    # Adds a new listener to this flow_manager. When a new listener is added it can
    # use the values of width, height and depth already initialized from the camera
    # ----------------------------------------------------------------------------------

    def add_listener(self, name, address):
        logging.info("%s: adding listener to flow_manager with name %s", self.video_name,
                     name)
        self._listeners[name] = address
        return (self.width, self.height, self.depth)
    
    # ----------------------------------------------------------------------------------
    # Removes the listener
    # ----------------------------------------------------------------------------------

    def remove_listener(self, name):
        logging.info("listener %s removed from flow_manager %s", name, self.video_name)
        del self._listeners[name]
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # CALLBACK METHODS
    
    # ----------------------------------------------------------------------------------
    # Callback method: when it becomes a listener to the video
    # decoder.  Only after the video decoder is initialize that we have information
    # about the width, height and depth of the video being decoded. 
    # ----------------------------------------------------------------------------------

    def initialize_mmap(self, width, height, depth):
        
        self.width = width
        self.height = height
        self.depth = depth
        self.frame_size = self.height * self.width * self.depth

        self._registered_trackers = len(self.trackers)

        # register the video with yolo.
        self.phone(self._yolo, 'register_video', self.video_name, self.video_id,
                   self.width, self.height, self.depth, callback = 'register_done')
        
        # open the mmap file with the decoded frames
        self._mmap = MmapFrames(self.video_name, self.width, self.height, self.depth)
        self._mmap.open_write2()

        # open the mmap file for communicating bounding boxes with yolo
        self._mmap_bbox = MmapBboxes()
        # self._mmap_bbox_buf = self._mmap_bbox.open_read()
        
        self._fix_dimensions()
        self._setting = Setting(self.cfg)
        
        # register the video with all trackers.  Need to wait for the registration
        # to be done to continues execution
        self._trackers_broadcast_with_callback(
            'register_video', self.video_name, self.video_id,
            self.width, self.height, self.depth, callback = 'register_done')
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def register_done(self, ret_value):
        self._registered_trackers -= 1
        # waiting for all trackers and also for yolo registration. Waiting for
        # termination of trackers +1 (yolo) process
        if self._registered_trackers < 0:
            # now that the mmap file has been initialized, we can call 'start_processing'
            # self.post(self.vd, 'start_processing')
            self.post(self.parent_address, 'flow_manager_initialized', self.video_name)
            
            self.proc_time = time.perf_counter()
            self._average = None
            
            # start an endless loop... process_frame calls many functions that end up
            # calling _next_frame, that call back process_frame
            self._process_frame()
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def continue_process(self):

        # total number of frames process by flow_manager.  This is not necessarily
        # equal to frame_number, as the video decoder might have dropped frames
        # when processing is not fast enought
        self._total_frames += 1
        
        # start the tracking phase
        self._tracking_phase()
                    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _check_bbox_int(self, bbox):
        p0 = bbox[0]
        p1 = bbox[1]
        p2 = bbox[2]
        p3 = bbox[3]

        if not isinstance(p0, int):
            logging.warning('p0 not int')
        if not isinstance(p1, int):
            logging.warning('p1 not int')
        if not isinstance(p2, int):
            logging.warning('p2 not int')
        if not isinstance(p3, int):
            logging.warning('p3 not int')

    # ----------------------------------------------------------------------------------
    # When tracking is done, the trackers calls back this method with the updated
    # items information
    # ----------------------------------------------------------------------------------

    def tracking_done(self, items_update):

        if not (items_update == None):
            del_items = []
            for item_id, update in items_update.items():
                confidence = update[0]
                bounding_box = update[1]
                self._setting.update(bounding_box)

                self._check_bbox_int(bounding_box)

                # check and remove all bounding boxes that have exited the setting.
                # Those that have not exited, should be updated
                exit = self._setting.check_exit(bounding_box)
                if exit:
                    # self._remove_item(item_id)
                    del_items.append(item_id)
                else:
                    self._setting.update_item(self.cfg.frame_number, item_id, confidence,
                                              bounding_box)
                self._remove_items(del_items)

        # are all trackers done? If all done then we can call the
        # detection phase
        self.num_trackers -= 1
        if self.num_trackers < 1:
            # -----------------------------
            # collecting metric information
            # -----------------------------
            # -----------------------------
            self._time_tracking += self._time_elapsed()
            # collecting metric information
            if self._total_frames % 100 == 0:
                logging.info("%s: average time of tracking is %f",
                             self.video_name, self._time_tracking / 100)
                self._total_time += self._time_tracking
                self._time_tracking = 0
            # -----------------------------
        # if self.cfg.is_image == True:
        #     self._write_metrics(self._setting.items)

        self._detection_phase()
            
    # ----------------------------------------------------------------------------------
    # Callback method for the 'find_bboxes' call to the Neural Net.  This callback is
    # registered by method 'process_frame'.
    # ----------------------------------------------------------------------------------

    def boxes_detected(self, boxes, confidences, classIDs):

        # -----------------------------
        # collecting metric information
        # -----------------------------
        # -----------------------------
        self._time_findbboxes += self._time_elapsed()
        # collecting metric information
        if self._total_frames % 100 == 0:
            logging.info("%s: average time running Yolo is %f",
                         self.video_name, self._time_findbboxes / 100)
            self._total_time += self._time_findbboxes
            self._time_findbboxes = 0
        # -----------------------------
        
        # convert the detected bounding boxes to Items
        self._setting.detections2items(boxes, confidences, classIDs)
        
        # add the newly detected items to the setting. This method will match the
        # already tracked items with the newly detected ones, adding only the
        # relevant items
        self._add_items()
        
        # -----------------------------
        # collecting metric information
        # -----------------------------
        # -----------------------------
        self._time_add_items += self._time_elapsed()
        # collecting metric information
        if self._total_frames % 100 == 0:
            logging.info("%s: average time adding items is %f",
                         self.video_name, self._time_add_items / 100)
            self._total_time += self._time_add_items
            self._time_add_items = 0
        # -----------------------------
        if self.cfg.is_image == True:
            self._write_metrics(self._setting.items)
        self._next_frame()
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS
            
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _time_elapsed(self):
        now = time.perf_counter()
        elapsed = (now - self.time_metric)
        self.time_metric = now
        return elapsed
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _process_frame(self):

        # initialize the time_metric at every new frame
        self.time_metric = time.perf_counter()

        self.post(self.vd, '_manage_buffer', self._average)
        self.frame_index += 1
        if self.frame_index == self._buffer_max_size - 1:
            self.frame_index = 0

        fn = 0
        while fn == 0 or fn < self.cfg.frame_number:
            fn = self._mmap.read_header(self.frame_index)

        # if (self.video_name == 'cshopp1'):
        #     logging.warning("%s: processing index %d with frame number: %d",
        #                     self.video_name, self.frame_index, fn)
            
        self.cfg.frame_number = fn
        
        logging.debug("******index %d: reading frame number %d ********",
                     self.frame_index, fn)
                            
        # self.continue_process(self._total_frames)
        self.continue_process()
        
    # ----------------------------------------------------------------------------------
    # Executes the tracking_phase of the algorithm.  Bascially calls method
    # update_tracked_items on tall the trackers.
    # ----------------------------------------------------------------------------------

    def _tracking_phase(self):

        # keep reference to the number of trackers that have already replied
        # with tracking information. None so far.
        self.num_trackers = len(self.trackers)

        # -----------------------------
        self._time_load += self._time_elapsed()
        # collecting metric information
        if self._total_frames % 100 == 0:
            logging.info("%s: average time to load frame is %f",
                         self.video_name, self._time_load / 100)
            self._total_time += self._time_load
            self._time_load = 0
        # -----------------------------
            
        # check for disappeared items and remove them:
        self._check_disappeared()
        
        # -----------------------------
        self._time_ckd += self._time_elapsed()
        # collecting metric information
        if self._total_frames % 100 == 0:
            logging.info("%s: average time of _check_disappeared is %f",
                         self.video_name, self._time_ckd / 100)
            self._total_time += self._time_ckd
            self._time_ckd = 0
        # -----------------------------
            
        # now drop overlaped items
        # logging.info(self._setting.find_overlap())
        self._remove_items(self._setting.find_overlap())
        
        # -----------------------------
        self._time_removal += self._time_elapsed()
        # collecting metric information
        if self._total_frames % 100 == 0:
            logging.info("%s: average time of removal is %f",
                         self.video_name, self._time_removal / 100)
            self._total_time += self._time_removal
            self._time_removal = 0
        # -----------------------------
            
        # do the tracking phase of the algorithm
        # update tracked items for this video every 'x' frames according to
        # configuration
        if (self.cfg.data['video_analyser']['track_every_x_frames'] == 1 or
            (self.cfg.frame_number %
             self.cfg.data['video_analyser']['track_every_x_frames'] == 0)):
            self._trackers_broadcast_with_callback(
                'update_tracked_items', self.video_name, self.frame_index,
                callback = 'tracking_done')
            
        # should always execute the detection phase. If doing a tracking phase
        # on the frame, then the call to detection_phase should be done after
        # all trackers have finished... this is done in the tracking_done method
        # bellow. If tracking is not done in the frame, then just call the
        # detection_phase directly
        else:
            logging.info("+++++++++++++++This should not be printed in this config++++++++++")
            self._detection_phase()

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def _detection_phase(self):
        # do detection on the frame
        if (self.cfg.frame_number >
            self._last_detection + self.cfg.data['video_analyser']['skip_detection_frames']):
            self._last_detection = self.cfg.frame_number
            logging.debug("%s: calling Yolo for frame %d", self.video_name,
                         self._total_frames)
            self.phone(self._yolo, 'find_bboxes', self.video_name, self.frame_index,
                       callback = 'boxes_detected')
        else:
            # This is one problem with callback functions: the '_next_frame' method is
            # called here and also on the 'boxes_detected' callback method. It's
            # a bit confusing.
            self._next_frame()
    
    # ----------------------------------------------------------------------------------
    # This method sends to the decoder the 'next_frame' message for it to
    # decode a new frame.  This closes the processing loop: 1) decoder decodes a frame;
    # 2) decoder calls 'process_frame' from flow_manager; 3) flow_manager does whatever
    # it needs to to with the frame; 4) flow_manager calls 'next_frame' (this method);
    # 5) 'next_frame' calls back onto the decoder (step 1 above)
    # ----------------------------------------------------------------------------------

    def _next_frame(self):

        # -----------------------------
        # collecting metric information
        # -----------------------------
        # -----------------------------
        self._time_detection += self._time_elapsed()
        # collecting metric information
        if self._total_frames % 100 == 0:
            logging.info("%s: average time of detection is %f",
                         self.video_name, self._time_detection / 100)
            self._total_time += self._time_detection
            self._time_detection = 0
        # -----------------------------
        
        # notify all the listeners to this flow_manager that we have finished
        # processing this frame and are going to process the next one.
        # TODO: Might not be enough time for all listeners to do something with the
        # frame before we get the next frame. Not a problem right now since we only
        # have one listener to this object
        self._notify_listeners()
        
        # -----------------------------
        # collecting metric information
        # -----------------------------
        # -----------------------------
        self._time_notif += self._time_elapsed()
        # collecting metric information
        if self._total_frames % 100 == 0:
            logging.info("%s: average time of notification is %f",
                         self.video_name, self._time_notif / 100)
            self._total_time += self._time_notif
            self._time_notif = 0
            avg_total_time = self._total_time / 100

            logging.info("%s: total average time for the whole process: %f",
                         self.video_name, avg_total_time)

        # -----------------------------

        # this is the time between the call to process_frame and _next_frame
        if self._total_frames % 100 == 0:
            now = time.perf_counter()
            self._average = (now - self.proc_time) / 100
            logging.info("%s: average processing time for the last 100 frames is %f",
                         self.video_name, self._average)
            logging.info("%s: time not accounted for is %f", self.video_name,
                         self._average - avg_total_time)
            self._total_time = 0
            self.proc_time = time.perf_counter()
            logging.info("===========================")

        # set the header of this frame to 0 indicating that this frame was processed
        self._mmap.write_header(self.frame_index, 0)
        # copy the current frame to the last position on the buffer that is never
        # used. This will be used by the 'display' and other listeners to work
        # with this frame
        self._mmap.copy_last(self.frame_index)
        
        # process the next frame
        self._process_frame()
        
    # ----------------------------------------------------------------------------------
    # broadcast a message to all the trackers.
    # ----------------------------------------------------------------------------------

    def _trackers_broadcast_with_callback(self, method, *args, **kwargs):
        for tracker_name, tracker in self.trackers.items():
            self.phone(tracker[0], method, *args, **kwargs)
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def _trackers_broadcast(self, method, *args, **kwargs):
        for tracker_name, tracker in self.trackers.items():
            self.post(tracker[0], method, *args, **kwargs)
        
    # ---------------------------------------------------------------------------------
    # 
    # ---------------------------------------------------------------------------------

    def _remove_items(self, items_ids):
        trackers = {}
        
        for item_id in items_ids:
            # The item might have been removed by going out of the entry lines
            if item_id in self._setting.items.keys():
                item = self._setting.items[item_id]
                key = str(item.tracker_address)
                if key not in trackers:
                    trackers[key] = {}
                    trackers[key]['doer_address'] = item.tracker_address
                    trackers[key]['items_ids'] = []
                    trackers[key]['items_ids'].append(item_id)
                del self._setting.items[item_id]

        for key in trackers:
            self.post(trackers[key]['doer_address'], 'stop_tracking_items',
                      self.video_name, trackers[key]['items_ids']) 
                              
    # ---------------------------------------------------------------------------------
    # 
    # ---------------------------------------------------------------------------------

    def _check_disappeared(self):

        if len(self._setting.items) == 0:
            return
        
        delete = []
        
        for item_id, item in self._setting.items.items():
            if (item.last_update != 0 and
                self.cfg.frame_number > item.last_update +
                self.cfg.data["trackable_objects"]["disappear"]):
                item.disappeared = True
                item.last_frame = self.cfg.frame_number
                delete.append(item_id)

        self._remove_items(delete)
                
    # ---------------------------------------------------------------------------------
    # Given a list of items to be tracked, send them for tracking to the multiple
    # trackers. This uses a very simple policy: breaks the items in chunks of 5 and
    # send every 5 items to a randomly selected tracker from the available trackers.
    # TODO: Try different policies for distributing items to trackers.
    # ---------------------------------------------------------------------------------

    def _distribute2trackers(self, items):

        chunck_size = 3
        
        final = [items[i * chunck_size:(i + 1) * chunck_size] for i in
                 range((len(items) + chunck_size - 1) // chunck_size )]

        for chunk in final:
            key = list(self.trackers.keys())[random.randrange(len(self.trackers))]
            # logging.info("%s: Selected tracker is %s", self.video_name, key)
            
            tracker = self.trackers[key]
            
            for item in chunk:
                # first frame where this item was detected
                item.first_frame = self.cfg.frame_number
                
                # set the id of this item to the next value
                self.next_item_id += 1
                item.item_id = self.next_item_id
                self._setting.items[self.next_item_id] = item
                item.tracker_address = tracker[0]

            self.post(tracker[0], 'tracks_list', self.video_name, self.frame_index,
                      items)
                
    # ---------------------------------------------------------------------------------
    # Matches the newly detected items with the already tracked items using either
    # "iou_match" or "centroid_match" algorithm.
    # match_row_cols are detected items that were already being tracked
    # unused_cols are new items
    # ---------------------------------------------------------------------------------

    def _match_items(self):
        
        if self.cfg.data["trackable_objects"]["match"] == "iou_match":
            (unused_rows,
             unused_cols,
             match_rows_cols) = Geom.iou_match(
                 list(self._setting.items.values()), self._setting.new_inputs,
                 self.cfg.data["trackable_objects"]["iou_match"])
        elif self.cfg.data["trackable_objects"]["match"] == "centroid_match":
            (unused_rows,
             unused_cols,
             match_rows_cols) = Geom.centroid_match(
                 list(self._setting.items.values()), self._setting.new_inputs,
                 self.cfg.data["trackable_objects"]["centroid_match_max_distance"])
        else:
            logging.info("Unknown matching algorithm specified")

        return (unused_rows, unused_cols, match_rows_cols)

    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    def _add_items(self):
        
        # if we are currently not tracking any objects we should
        # start tracking them
        if (len(self._setting.items) == 0):
            self._distribute2trackers(self._setting.new_inputs)
            return

        # match the new items to the already tracked objects using the matching
        # algorithgm in the configuration file
        (unused_rows, unused_cols, match_rows_cols) = self._match_items()

        new_inputs = [self._setting.new_inputs[col] for col in unused_cols]
        self._distribute2trackers(new_inputs)
        
    # ---------------------------------------------------------------------------------
    # This method notifies all listeners that we have a new frame processed. It sends
    # the following messages to the listeners:
    # * 'base_image': with the size of the buffer (mmap) where the image is
    # * 'overlay_bboxes': with all the detection boxes found
    # * 'add_lines': to add the entry_lines
    # * 'add_lines': to add the counting_lines
    # * 'display': tells the listener that it can display the image
    # ---------------------------------------------------------------------------------

    def _notify_listeners(self):

        # notify every listener that we have a new frame and give it the
        # buffer size
        for name, listener in self._listeners.items():
            # listener: doer's address
            # when sending the base image, send also all the items, so that they
            # can be used by other methods
            self.post(listener, 'base_image', list(self._setting.items.values()))
            self.post(listener, 'overlay_bboxes')
            self.post(listener, 'add_id')
            self.post(listener, 'add_lines', self.cfg.data['entry_lines'])
            self.post(listener, 'add_lines', self.cfg.data['counting_lines'], True)
            self.post(listener, 'display')
        
    # ----------------------------------------------------------------------------------
    # Dimension configurations (on the configuration file) are done over an image of
    # a certain dimension.  If we show the image in another dimension, the dimensions
    # need to be converted to the new dimension
    # ----------------------------------------------------------------------------------

    def _fix_dimensions(self):
        
        lines_dimensions = self.cfg.data['video_processor']['lines_dimensions']
        
        # Constants needed to resize the identified bboxes to the original frame size
        kw = self.width/lines_dimensions[0]
        kh = self.height/lines_dimensions[1]

        for line_name, spec in self.cfg.data['entry_lines'].items():
            end_points = spec['end_points']
            spec['end_points'] = [int(end_points[0] * kw), int(end_points[1] * kh),
                                  int(end_points[2] * kw), int(end_points[3] * kh)]

        for line_name, spec in self.cfg.data['counting_lines'].items():
            end_points = spec['end_points']
            spec['end_points'] = [int(end_points[0] * kw), int(end_points[1] * kh),
                                  int(end_points[2] * kw), int(end_points[3] * kh)]

            # fix the dimension of the counter's position
            label1_position = spec['label1_position']
            label2_position = spec['label2_position']
            spec['label1_position'] = [int(label1_position[0] * kw),
                                       int(label1_position[1] * kh)]
            spec['label2_position'] = [int(label2_position[0] * kw),
                                       int(label2_position[1] * kh)]

        
        self.cfg.data['video_processor']['lines_dimensions'] = [self.width,
                                                                self.height]
                                  
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _add_listener(self, v2):
        # Starts displaying the video on a new window. For this, add a new listener
        # to the video_decoder and have it callback the initialize method of the
        # Display we have just created above
        self.add_listener(self.video_name, self._dp)
        self.playback_started = True


    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    # DEPRECATED
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def _remove_item2(self, item_id):

        # The item might have been removed by going out of the entry lines
        if item_id not in self._setting.items.keys():
            return
        
        item = self._setting.items[item_id]
        del self._setting.items[item_id]
        self.post(
            item.tracker_address, 'stop_tracking', self.video_name, item_id)
    
    # ---------------------------------------------------------------------------------
    # 
    # ---------------------------------------------------------------------------------

    def _remove_items2(self, items):
        for item in items:
            self._remove_item(item)
        
    def _write_metrics(self, items_update):
        metric_file = open(self.cfg.data['io']['record'] + ".metrics", "w")
        for item_id, update in items_update.items():
            confidence = update.confidence
            startx = update.startX
            starty = update.startY
            endx = update.endX
            endy = update.endY
            metric_file.write("%d, %d, %f, %f, %f, %f, %f, -1, -1, -1\n" %
                              (self.cfg.frame_number, item_id, startx, starty, endx-startx, endy-starty, confidence))
        metric_file.close()