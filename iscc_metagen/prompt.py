# -*- coding: utf-8 -*-
import functools
import inspect
import textwrap
from jinja2 import Environment, StrictUndefined


def make_prompt(fn) -> str:
    """Decorate a function that contains a prompt template.

    This allows you to define prompts in the docstring of a function and render them using Jinja2
    with the function's arguments.
    """
    template = textwrap.dedent(fn.__doc__)
    if template is None:
        raise TypeError("Could not find a template in the function's docstring.")

    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        env = Environment(
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )
        jinja_template = env.from_string(template)
        return jinja_template.render(**bound_arguments.arguments)

    return wrapper
