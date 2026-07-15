import importlib.util
import os
import sysconfig
from pathlib import Path
from typing import Optional


_stdlib_logging_path = os.path.join(sysconfig.get_path("stdlib"), "logging", "__init__.py")
_stdlib_logging_spec = importlib.util.spec_from_file_location("_stdlib_logging", _stdlib_logging_path)
_stdlib_logging = importlib.util.module_from_spec(_stdlib_logging_spec)
_stdlib_logging_spec.loader.exec_module(_stdlib_logging)

for _name in dir(_stdlib_logging):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_stdlib_logging, _name)


def get_logger(name: str, log_file: Optional[str] = None) -> Logger:
    logger = getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(INFO)
    formatter = Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    stream_handler = StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
