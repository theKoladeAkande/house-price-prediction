import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import os
import sys


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_api.log'



FOMARTER = logging.Formatter(
                    "%(asctime)s-%(name)s-%(levelname)s-"
                    "%(funcName)s:%(lineno)d-%(message)s")



def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FOMARTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when = "midnight")
    file_handler.setFormatter(FOMARTER)
    file_handler.setLevel(logging.INFO)
    return file_handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())

    logger.propagate = False

    return logger


class Config: 
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = False    
    SECRET_KEY = 'changed-in-place'
    SEVER_PORT = 5000


class ProductionConfig(Config):
    DEBUG = False
    SEVER_PORT = os.environ.get('PORT', 5000)


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True