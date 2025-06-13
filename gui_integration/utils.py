

import sys
import logging
import coloredlogs


LOG_LEVEL = logging.DEBUG
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=LOG_LEVEL, stream= sys.stdout)
    coloredlogs.install(level=LOG_LEVEL, stream= sys.stdout)
    return logging.getLogger(name)

