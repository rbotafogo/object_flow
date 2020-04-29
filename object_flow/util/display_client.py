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
import numpy as np

import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='display')

def callback(ch, method, properties, body):
    video_name = properties.headers['video_name']
    nparr = np.fromstring(body, np.uint8)
    dec = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    cv2.imshow("Object Flow - " + video_name, dec)
    cv2.waitKey(25)
    # print(" [x] Received %r" % body)

channel.basic_consume(
    queue='display', on_message_callback=callback, auto_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')

channel.start_consuming()
