import logging
import sys
from typing import Optional


def get_logger(
    name: str = "llm_eval",
    level: int = logging.INFO,
    log_to_stdout: bool = True,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """
    A simple logger configuration function.

    Args:
        name (str): Logger name (usually the package or module name)
        level (int): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_stdout (bool): If True, output to stdout. If False,
            do not add a handler.
        log_format (str): Custom log format. If None, the default format is used.

    Returns:
        logging.Logger: Configured logger object
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers (to avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    if not log_format:
        log_format = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

    formatter = logging.Formatter(log_format)

    # STDOUT handler
    if log_to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # Optionally, file logging can be added if needed:
    # file_handler = logging.FileHandler("llm_eval.log")
    # file_handler.setLevel(level)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger
