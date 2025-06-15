

import sys
import logging
import coloredlogs


LOG_LEVEL = logging.DEBUG
def get_logger(name: str) -> logging.Logger:
    fmt_str = "{asctime} {name} {levelname} {message}"
    logging.basicConfig(level=LOG_LEVEL, stream= sys.stdout, format=fmt_str, style='{')
    coloredlogs.install(level=LOG_LEVEL, stream= sys.stdout, fmt=fmt_str, style='{')
    return logging.getLogger(name)

