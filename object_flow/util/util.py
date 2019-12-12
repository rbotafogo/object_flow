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

from datetime import datetime
from datetime import timedelta
from pytz import timezone
from absl import logging
from calendar import monthrange

class Util:
    
    # ---------------------------------------------------------------------------------
    # Returns the current datetime in São Paulo time zone formatted in a
    # day_month_year_hour:minute:second format
    # ---------------------------------------------------------------------------------

    @classmethod
    def br_datetime(cls):
        now = datetime.now()
        tz = timezone('America/Sao_Paulo')
        now_br = now.astimezone(tz)
        return now_br.strftime('%d_%m_%Y_%H:%M:%S')
        
    # ---------------------------------------------------------------------------------
    # Return the current datetime in São Paulo time zone formatted in a
    # yearmonthday_hourminutesecond format
    # ---------------------------------------------------------------------------------

    @classmethod
    def brus_datetime(cls):
        now = datetime.now()
        tz = timezone('America/Sao_Paulo')
        now_br = now.astimezone(tz)
        return now_br.strftime('%Y%m%d_%H%M%S')
        
    # ---------------------------------------------------------------------------------
    # Return the current time in the São Paulo time zone in hour:minute:second format
    # ---------------------------------------------------------------------------------

    @classmethod
    def br_time(cls):
        now = datetime.now()
        tz = timezone('America/Sao_Paulo')
        now_br = now.astimezone(tz)
        return now_br.strftime('%H:%M:%S')
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    @classmethod
    def br_time_raw(cls):
        now = datetime.now()
        tz = timezone('America/Sao_Paulo')
        return now.astimezone(tz)
        
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    @classmethod
    def to_br_time(cls, dt):
        print(dt)
        tz = timezone('America/Sao_Paulo')
        return dt.astimezone(tz)

    # ---------------------------------------------------------------------------------
    # rounds up the date time parameter to the given delta value.  For example, if it
    # is dt = 10:12 and delta is 10, round_dt will return 10:20; if delta = 30,
    # round_dt will return 10:30
    # ---------------------------------------------------------------------------------

    @classmethod
    def round_dt(cls, dt, delta):
        minute = dt.minute
        hour = dt.hour
        round_minute = (minute // delta * delta + delta) % 60
        if round_minute == 0:
            round_minute = 0
            hour += 1
        round_dt = dt.replace(hour=hour, minute=round_minute, second=0)        
        return round_dt
    
    # ---------------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------------

    @classmethod
    def set_tzaware_time(cls, hour = 0, minute = 0, second = 0, delta_day = 0):
        now = Util.br_time_raw()
        delta = timedelta(days=delta_day)
        next_date = now + delta
        month = next_date.month
        year = next_date.year
        next_date = next_date.replace(year=year, month=month, hour=hour, minute=minute,
                                      second=second)
        return next_date
    
    # ----------------------------------------------------------------------------------
    #
    # ----------------------------------------------------------------------------------

    @classmethod
    def isoweekday(cls, dt = None):
        if dt == None:
            dt = Util.br_time_raw()
        return dt.isoweekday()

