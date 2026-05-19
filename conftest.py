import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.logger import setup_logger


def pytest_configure(config):
    setup_logger(
        level=logging.DEBUG,
        log_dir="logs",
        log_file="test.log",
    )
logging.getLogger("DERMIGHT").propagate = True