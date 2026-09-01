from uuid import uuid4

__logger_filename = f"{str(uuid4())}.log"
__is_set = False
__is_set_logger = False
__logger = None 

def set_logfilename(filename, override=False):
    global __is_set
    global __logger_filename
    if not __is_set or override:
        __logger_filename = filename
        __is_set = True

def get_logfilename():
    return __logger_filename


def set_logger(logger, override=False):
    global __logger
    global __is_set_logger
    if not __is_set_logger or override:
        __logger = logger
        __is_set_logger = True

def get_logger():
    return __logger