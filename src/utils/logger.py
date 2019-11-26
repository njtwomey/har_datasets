# Adapted from:
#   https://gist.github.com/nguyenkims/e92df0f8bd49973f0c94bddf36ed7fd0

import logging
import sys
from logging import FileHandler

FORMATTER = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_FILE = "logging.log"


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = FileHandler(LOG_FILE, mode="w")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def logger_exception(logger):
    str_exception = logger.exception
    
    def exception(msg, *args, exc_info=True, **kwargs):
        if not isinstance(msg, Exception):
            logger.warn(
                f'logger.exception called outside of Exception scope'
            )
            msg = Exception(msg)
        try:
            raise msg
        except Exception as ex:
            str_exception(msg, *args, exc_info=exc_info, **kwargs)
            raise ex
    
    return exception


def get_logger(logger_name, with_file=True, with_console=True, raise_exceptions=True):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    if with_console:
        logger.addHandler(get_console_handler())
    if with_file:
        logger.addHandler(get_file_handler())
    logger.propagate = False
    
    if raise_exceptions:
        logger.exception = logger_exception(logger)
    
    return logger
