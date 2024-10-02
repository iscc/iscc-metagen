"""Utility functions."""

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
