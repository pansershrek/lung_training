import logging


class Logger(object):
    def __init__(self, log_level, logger_name, log_file_name=None):
        # firstly, create a logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        # secondly, create a handler
        file_handler=None
        if log_file_name: file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        # thirdly, define the output form of handler
        formatter = logging.Formatter(
            '[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s '
        )
        if file_handler:file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # finally, add the Hander to logger
        if file_handler:self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger