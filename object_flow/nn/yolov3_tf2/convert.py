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

from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights

from contagem.ui.util import Util

flags.DEFINE_string('weights', './data/yolov3.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov3.tf', 'path to output')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.summary()
    logging.info(Util.br_time() + ', model created')

    load_darknet_weights(yolo, FLAGS.weights, FLAGS.tiny)
    logging.info(Util.br_time() + ', weights loaded')

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    logging.info(Util.br_time() + ', sanity check passed')

    yolo.save_weights(FLAGS.output)
    logging.info(Util.br_time() + ', weights saved')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
