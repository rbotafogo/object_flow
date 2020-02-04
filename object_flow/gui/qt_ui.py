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

from PyQt5.QtWidgets import (
    QWidget, QToolTip, QPushButton, QApplication, QInputDialog, QLineEdit)
from PyQt5.QtCore import QThreadPool
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSlot

import logging

from object_flow.ipc.board import Board
from object_flow.flow.multi_flow import MultiFlow

class CountingGUI(QWidget):
    
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self._default_log()
        self.initUI()

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    @pyqtSlot()
    def start_click(self):        
        self.board = Board(proc = 'udp', logcfg=self.logcfg)
        self.board.hire('MultiFlow', MultiFlow, self.cfg)

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    @pyqtSlot()
    def quit_click(self):
        self.board.tell('MultiFlow', 'terminate')
        self.board.shutdown()
        QApplication.instance().quit()
        
        
    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    @pyqtSlot()
    def playback_click(self):
        self.board.tell('MultiFlow', 'start_playback', 'c9')

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------
    
    def initUI(self):

        QToolTip.setFont(QFont('SansSerif', 10))
        
        sbtn = QPushButton('Start', self)
        sbtn.setToolTip('Starts analysing all the videos')
        sbtn.resize(sbtn.sizeHint())
        sbtn.move(50, 50)

        qbtn = QPushButton('Quit', self)
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(50, 100)

        # text, okPressed = QInputDialog.getText(
        #     self, "Playback camera","Camera name:", QLineEdit.Normal, "")
        # if okPressed and text != '':
        #     print(text)
            
        # pbtn = QPushButton('Playback', self)
        # pbtn.resize(qbtn.sizeHint())
        # pbtn.move(50, 150)

        sbtn.clicked.connect(self.start_click)
        qbtn.clicked.connect(self.quit_click)
        # pbtn.clicked.connect(self.playback_click)
        
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Object Flow - Command Pannel')    
        self.show()        

    # ----------------------------------------------------------------------------------
    # 
    # ----------------------------------------------------------------------------------

    def _default_log(self):
        
        self.logcfg = { 'version': 1,
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
        
