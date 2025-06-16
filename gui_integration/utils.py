

import sys
import logging
import coloredlogs


LOG_LEVEL = logging.DEBUG
def get_logger(name: str) -> logging.Logger:
    fmt_str = "{asctime} {name} {levelname} {message}"
    logging.basicConfig(level=LOG_LEVEL, stream= sys.stdout, format=fmt_str, style='{')
    coloredlogs.install(level=LOG_LEVEL, stream= sys.stdout, fmt=fmt_str, style='{')
    return logging.getLogger(name)


def prob_to_color(prob: float) -> str:
    color1 = (0.0, 1.0, 0.0)
    color2 = (1.0, 0.0, 0.0)
    mixed = (
        int(0xFF * pow(color1[0] * (1 - prob) + color2[0] * prob, 1.0 / 2.2)),
        int(0xFF * pow(color1[1] * (1 - prob) + color2[1] * prob, 1.0 / 2.2)),
        0x00
    )
    return "#{:02x}{:02x}00".format(mixed[0], mixed[1])

