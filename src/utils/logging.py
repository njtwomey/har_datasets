# Adapted from:
#   https://gist.github.com/nguyenkims/e92df0f8bd49973f0c94bddf36ed7fd0

import logging
import sys
from logging import FileHandler

FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_FILE = "logging.log"


def get_console_handler():
    """
    
    Returns:

    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    """
    
    Returns:

    """
    file_handler = FileHandler(LOG_FILE)
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name, with_file=True, with_console=False):
    """
    
    Args:
        logger_name:
        with_file:
        with_console:

    Returns:

    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    if with_console:
        logger.addHandler(get_console_handler())
    if with_file:
        logger.addHandler(get_file_handler())
    logger.propagate = False
    return logger
