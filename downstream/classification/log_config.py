# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành (@bu1th4nh)
# Title: log_config.py
# Date: 2024/09/24 10:04:59
# Description: Log configuration
# 
# (c) bu1th4nh. All rights reserved
# -----------------------------------------------------------------------------------------------


import numpy as np
import pandas as pd

import logging
import sys
from colorlog import ColoredFormatter


# -----------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------
def initialize_logging(log_filename: str = "crossomic.log"):
    # logging.root = logging.getLogger('uvicorn.default')
    logging.root.handlers = [];

    # Console handler
    handler_sh = logging.StreamHandler(sys.stdout)
    handler_sh.setFormatter(
        ColoredFormatter(
            "%(cyan)s%(asctime)s.%(msecs)03d %(log_color)s[%(levelname)s]%(reset)s %(light_white)s%(message)s%(reset)s %(blue)s(%(filename)s:%(lineno)d)",
            datefmt  = '%Y/%m/%d %H:%M:%S',
            log_colors={
                'DEBUG': 'white',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )



    # Set logging level and handlers
    logging.basicConfig(
        level    = logging.INFO,
        handlers = [handler_sh]
    )



