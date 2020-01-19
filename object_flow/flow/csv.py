 # -*- coding: utf-8 -*-

##########################################################################################
# @author Rodrigo Botafogo
#
# Copyright (C) 2019 Rodrigo Botafogo - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Rodrigo Botafogo <rodrigo.a.botafogo@gmail.com>, 2019
##########################################################################################

import logging
import os
import csv
import datetime

from object_flow.util.util import Util

# =========================================================================================
#
# =========================================================================================

class CSV:

    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------
    
    def initialize(cfg):
        now = Util.br_time_raw()
        
        if (cfg.start_time > now):
            cfg.delta_csv_update = datetime.timedelta(
                seconds=((cfg.start_time - now).total_seconds()))
        else:
          cfg.delta_csv_update = datetime.timedelta(
                minutes=cfg.minutes)
        
        logging.info("delta_csv_update is %s", str(cfg.delta_csv_update))
        
        cfg.last_csv_update = Util.br_time_raw()
        
    # ----------------------------------------------------------------------------------
    # delta_time is given in minutes
    # ----------------------------------------------------------------------------------

    def csv_schedule(cfg):
        now = Util.br_time_raw()

        if (now >= cfg.last_csv_update + cfg.delta_csv_update):
            # read the global system config file for the next delta.
            # TODO: This value could be changes on the fly by an operator
            cfg.delta_csv_update = datetime.timedelta(
                minutes=cfg.system_info['minutes'])
            logging.info("timedelta for next csv update is: %s",
                         str(cfg.delta_csv_update))
            cfg.last_csv_update = now
            CSV._append_csv(cfg)

    # ----------------------------------------------------------------------------------
    # Appends data to the csv file for analytics
    # ----------------------------------------------------------------------------------

    def _append_csv(cfg):
        
        now = Util.br_time_raw()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute

        cfg.file_path = cfg.output_dir + "/" + Util.brus_datetime() + "_" + cfg.file_name
        
        if cfg.system_info['with_min']:
            row = ['Loja', 'Id', 'Ano', 'Mes', 'Dia', 'Hora', 'Minuto', 'Contagem']
        else:
            row = ['Loja', 'Id', 'Ano', 'Mes', 'Dia', 'Hora', 'Contagem']
            
        with open(cfg.file_path + ".csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
            for line_name, spec in cfg.data["counting_lines"].items():
                row1 = []
                row2 = []
                # First label
                row1.append(line_name + "-" + spec['label1_text'])
                row1.append(spec['label1_id'])
                row1.append(year)
                row1.append(month)
                row1.append(day)
                row1.append(hour)
                if cfg.system_info['with_min']:
                    row1.append(minute)
                row1.append(spec['counter1'])
                # Second Label
                row2.append(line_name + "-" + spec['label2_text'])
                row2.append(spec['label2_id'])
                row2.append(year)
                row2.append(month)
                row2.append(day)
                row2.append(hour)
                if cfg.system_info['with_min']:
                    row2.append(minute)
                row2.append(spec['counter2'])
                spec['counter1'] = 0
                spec['counter2'] = 0
                
                writer.writerow(row1)
                writer.writerow(row2)
            
