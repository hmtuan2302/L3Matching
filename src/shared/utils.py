from __future__ import annotations

import hashlib
import html
import os
import re
import time
from functools import lru_cache, wraps

from shared.logging import get_logger
from shared.settings import Settings

@lru_cache
def get_settings():
    return Settings()  # type: ignore

logger = get_logger(__name__)

def clean_str(input: str) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1]
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', result)


def compute_sha1_from_file(file_path: str) -> str:
    """Compute sha1 code of file from file path

    Args:
        file_path (str): file path

    Returns:
        str: sha1 code
    """
    with open(file_path, 'rb') as file:
        code = file.read()
        readable_hash = compute_sha1_from_content(code)

    return readable_hash


def compute_sha1_from_content(content: bytes) -> str:
    """Compute sha1 from content in the form of byte

    Args:
        content (bytes): content in the form of byte

    Returns:
        str: hash code content in the form of string
    """
    readable_hash = hashlib.sha1(content).hexdigest()

    return readable_hash


def profile(func):
    """Decorator to profile execution time. Using default logger with info level\n
    Output: [module.function] executed in: 0.0s
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger = get_logger('profiler')
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(
            f'[{func.__module__}.{func.__name__}] executed in: {end_time - start_time}s',
        )

        if hasattr(result, 'processing_time'):
            setattr(result, 'processing_time', end_time - start_time)

        return result

    return wrapper


def get_file_extension(file_name: str) -> str:
    """Get file extension from file name

    Args:
        file_name (str): file name

    Returns:
        str: file extension
    """
    return os.path.splitext(file_name)[-1].lower()
