"""Utility functions."""

import time
from loguru import logger as log
from iscc_metagen.settings import mg_opts
from functools import cache
from litellm import get_max_tokens, token_counter


def count_tokens(text, model_name=mg_opts.litellm_model_name):
    # type: (str, str) -> int
    return token_counter(model=model_name.split(":")[0], text=text)


@cache
def max_tokens(model_name=mg_opts.litellm_model_name, trim_ratio=0.75):
    # type: (str, float) -> int
    """Get context limit for currently configured model (if available)"""
    mt = get_max_tokens(model_name.split(":")[0]) or 4096
    mt = int(mt * trim_ratio)
    return mt


class timer:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        # Record the start time
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        # Calculate the elapsed time
        elapsed_time = time.perf_counter() - self.start_time
        # Log the message with the elapsed time
        log.debug(f"{self.message} {elapsed_time:.4f} seconds")
