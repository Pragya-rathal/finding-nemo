import logging
from pathlib import Path

def create_logger(name: str, out_dir: str) -> logging.Logger:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger=logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt=logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh=logging.FileHandler(Path(out_dir)/f'{name}.log')
    sh=logging.StreamHandler()
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger
