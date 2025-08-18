import logging


def get_log_level_from_str(level_str: str) -> int:
    level_str = level_str.upper()
    if level_str == "DEBUG":
        return logging.DEBUG
    elif level_str == "INFO":
        return logging.INFO
    elif level_str == "WARNING":
        return logging.WARNING
    elif level_str == "ERROR":
        return logging.ERROR
    elif level_str == "CRITICAL":
        return logging.CRITICAL
    else:
        raise ValueError(f"Unknown log level: {level_str}")


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Logs are of form <timestamp> <level> <filename>:<lineno> <message>
    """
    logger = logging.getLogger(name)
    logger.setLevel(get_log_level_from_str(level))

    handler = logging.StreamHandler()
    handler.setLevel(get_log_level_from_str(level))

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
