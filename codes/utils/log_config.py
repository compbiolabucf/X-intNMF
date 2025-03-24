# /*==========================================================================================*\
# **                        _           _ _   _     _  _         _                            **
# **                       | |__  _   _/ | |_| |__ | || |  _ __ | |__                         **
# **                       | '_ \| | | | | __| '_ \| || |_| '_ \| '_ \                        **
# **                       | |_) | |_| | | |_| | | |__   _| | | | | | |                       **
# **                       |_.__/ \__,_|_|\__|_| |_|  |_| |_| |_|_| |_|                       **
# \*==========================================================================================*/


# -----------------------------------------------------------------------------------------------
# Author: Bùi Tiến Thành - Tien-Thanh Bui (@bu1th4nh)
# Title: log_config.py
# Date: 2025/03/24 17:15:28
# Description: 
# 
# (c) 2025 bu1th4nh / UCF Computational Biology Lab. All rights reserved. 
# Written with dedication in the University of Central Florida, EPCOT and the Magic Kingdom.
# -----------------------------------------------------------------------------------------------



import sys
import logging
from typing import Union
from colorlog import ColoredFormatter

# -----------------------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------------------
def initialize_logging(log_filename: Union[str, None]):
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
    

    # File handler
    if log_filename is not None:
        handler_file = logging.FileHandler(
            filename = log_filename,
            encoding = "utf-8-sig",    
            mode     = "w"
        )
        handler_file.setFormatter(
            logging.Formatter(
                fmt      = '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)', 
                datefmt  = '%Y/%m/%d %H:%M:%S'
            )
        )

        # Set logging level and handlers
        logging.basicConfig(
            level    = logging.INFO,
            handlers = [handler_sh, handler_file]
        )
    else:
        logging.basicConfig(
            level    = logging.INFO,
            handlers = [handler_sh]
        )




