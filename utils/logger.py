import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


_LOGGER_INITIALIZED = False


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[35m", # Magenta
    }
    
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"
    
def setup_logger(
    name: str = "DERMIGHT",
    log_dir: str = "logs",
    log_file: str = "app.log",
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)
    logger.propagate = False
    
    if _LOGGER_INITIALIZED:
        return logger
    
    log_format = (
         "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    
    data_format  = "%Y-%m-%d %H:%M:%S"
    plain_formatter = logging.Formatter(
        fmt=log_format,
        datefmt=data_format,
    )
    colored_formatter = ColoredFormatter(
        fmt=log_format,
        datefmt=data_format,
    )
    
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)

    if file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=Path(log_dir) / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

    _LOGGER_INITIALIZED = True
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:

    base_logger_name = "DERMIGHT"

    if name is None:
        return logging.getLogger(base_logger_name)

    return logging.getLogger(f"{base_logger_name}.{name}")


def set_log_level(level: int) -> None:

    logger = logging.getLogger("DERMIGHT")
    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)