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

import cv2
import logging
import numpy as np
import pika
import json

from object_flow.util.mmap_frames import MmapFrames
from object_flow.ipc.doer import Doer

#==========================================================================================
#
#==========================================================================================

class Display(Doer):

    # ----------------------------------------------------------------------------------
    # When Display is create the initialize method should be called first.
    # @param video_name [String] name of the video camera
    # ----------------------------------------------------------------------------------

    def __initialize__(self, video_name, cfg):
        self.video_name = video_name
        self.cfg = cfg
        self._stop = False

        if cfg.message_broker:
            credentials = pika.PlainCredentials(
                cfg.mbroker['user'],
                cfg.mbroker['passwd'])
            parameters = pika.ConnectionParameters(
                cfg.mbroker['IPAddress'],
                cfg.mbroker['port'],
                cfg.mbroker['directory'],
                credentials)
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            self.channel.queue_declare(
                queue=cfg.mbroker['display_queue'])
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def terminate(self):
        super().terminate()
        self._mmap.close()
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # SERVICES
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def base_image(self, items):
        if self._stop:
            return

        self.items = items
        # header, self.frame = self._mmap.read_data(frame_index)
        header, self.frame = self._mmap.read_last()
        
    # ----------------------------------------------------------------------------------
    # overlay the bounding boxes on the frame. If centroids = True then add also the
    # bounding box centroid to the image
    # ----------------------------------------------------------------------------------
    
    def overlay_bboxes(self):
        if self.cfg.data['video_processor']['show_tracking_bbox'] == False:
            return

        color = self.cfg.data['video_processor']['tracking_bbox_color']
        for item in self.items:
            logging.debug((item.startX, item.startY, item.endX, item.endY))
            cv2.rectangle(self.frame, (item.startX, item.startY),
                          (item.endX, item.endY), color, 2)
    
    # ---------------------------------------------------------------------------------
    # Add the item id to the image
    # ---------------------------------------------------------------------------------

    def add_id(self):
        if self.cfg.data['video_processor']['show_id'] == False:
            return

        color = self.cfg.data['video_processor']['id_color']
        
        for item in self.items:
            text = "{}".format(item.item_id)
            if (item.cX < 10):
                bbx = 0
                item.cX = 5
            else:
                bbx = item.cX - 10
            if (item.cY < 10):
                bby = 0
                item.cY = 5
            else:
                bby = item.cY - 10

            if (item.cX >= self.width):
                bbx = self.width - 10
                item.cX = bbx
            else:
                bbx = item.cX - 10
            if (item.cY >= self.height):
                bby = self.height - 10
                item.cY = bby
            else:
                bby = item.cY - 10
            
            cv2.putText(self.frame, text, (bbx, bby),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            cv2.circle(self.frame, (item.cX, item.cY), 4, color, -1)

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def add_lines(self, lines, counting = False):
        
        for line_name, spec in lines.items():
            end_points = spec['end_points']
            first_point = (int(end_points[0]), int(end_points[1]))
            second_point = (int(end_points[2]), int(end_points[3]))

            cv2.line(self.frame, first_point, second_point, spec['line_color'], 2)
            
            if counting:
                self._add_counters(spec)

    # ----------------------------------------------------------------------------------
    # Sends the image through the rabbitmq queue
    # ----------------------------------------------------------------------------------

    def display(self):

        if self.cfg.message_broker:
            # convert image to string so that it can be sent through MQ
            img_str = cv2.imencode('.jpg', self.frame)[1].tostring()

            # send the actual frame
            self.channel.basic_publish(
                exchange='', routing_key = 'display',
                properties = pika.BasicProperties(
                    headers = {'video_name': self.video_name}),
                body = img_str)
        else:
            cv2.imshow("Iris 8 - Contagem - " + self.video_name, self.frame)
            cv2.setMouseCallback("Iris 8 - Contagem - " + self.video_name,
                                 self._read_input, self.video_name)
            cv2.waitKey(25)

    # ----------------------------------------------------------------------------------
    # Sends the image through the rabbitmq queue
    # ----------------------------------------------------------------------------------

    def display2(self):

        if self.cfg.message_broker:
            # convert image to string so that it can be sent through MQ
            img_str = cv2.imencode('.jpg', self.frame)[1].tostring()

            # send the actual frame
            self.channel.basic_publish(
                exchange='', routing_key = 'display',
                properties = pika.BasicProperties(
                    headers = {'video_name': self.video_name}),
                body = img_str)
        else:
            cv2.imshow("Iris 8 - Contagem - " + self.video_name, self.frame)
            cv2.setMouseCallback("Iris 8 - Contagem - " + self.video_name,
                                 self._read_input, self.video_name)
            cv2.waitKey(25)

    # ----------------------------------------------------------------------------------
    # Destroys the display window. Need to set self._stop = True to make sure that no
    # other frame will be shown, which would make the window reapear.
    # ----------------------------------------------------------------------------------

    def destroy_window(self):
        self._stop = True
        cv2.destroyAllWindows()
        self.connection.close()
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # CALLBACK METHODS
    
    # ----------------------------------------------------------------------------------
    # Callback method needed to initialize the mmap file.  This could be called by
    # any mmap file generator.
    # @param mmap_path [String] path of the mmap file
    # @param width [Integer] width of the image in the mmap file
    # @param height [Integer] height of the image in the mmap file
    # @param depth [Integer] depth of the image in the mmap file
    # ----------------------------------------------------------------------------------

    def initialize_mmap(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
        self.frame_size = width * height * depth

        self._mmap = MmapFrames(self.video_name, width, height, depth)
        self._mmap.open_read()
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    # PRIVATE METHODS
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _draw_counter(self, direction, value, counter, color):
        text = direction + ": {}".format(value)
        
        cv2.putText(self.frame, text, counter,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def _add_counters(self, spec):
        
        pos1 = (spec["label1_position"][0], spec["label1_position"][1])
        pos2 = (spec["label2_position"][0], spec["label2_position"][1])
        
        self._draw_counter(spec["label1_text"], spec["counter1"], pos1,
                           spec["label1_color"])
        self._draw_counter(spec["label2_text"], spec["counter2"], pos2,
                           spec["label2_color"])
        
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    def _read_input(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = (x, y)
            logging.info("Camera %s: %s", param, str(self._fix_dimensions(refPt)))

    # ----------------------------------------------------------------------------------
    # Dimension configurations (on the configuration file) are done over an image of
    # a certain dimension.  If we show the image in another dimension, the dimensions
    # need to be converted to the new dimension
    # ----------------------------------------------------------------------------------

    def _fix_dimensions(self, point):

        x = point[0]
        y = point[1]
        
        lines_dimensions = self.cfg.data['video_processor']['lines_dimensions']
        
        # Constants needed to resize the identified bboxes to the original frame size
        kw = 416/self.width
        kh = 416/self.height

        xprime = x * kw
        yprime = y * kh
        
        return (int(xprime), int(yprime))

    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------

        # for pika communication
        # credentials = pika.PlainCredentials('CountingApp', 'test1234')

        # Using remote port forwarding with SSH. Connection with the
        # server should be done as:
        #       ssh -X -R 8080:$SSH_CLIENT:5672
        # this makes 'localhost' on port 8080 to be forwarded to the
        # SSH_CLIENT on port 5672 where RabbitMQ is running
        # parameters = pika.ConnectionParameters('localhost',
        #                                        8080,
        #                                        '/',
        #                                        credentials)

        # self.connection = pika.BlockingConnection(parameters)
        # self.channel = self.connection.channel()
        # self.channel.queue_declare(queue='display')
    
