import numpy as np


def to_tuple(o):
    if isinstance(o, (tuple, list)):
        return tuple(o)
    else:
        return o,


def with_defaults(options: dict, defaults_dict: dict=None, **defaults_kwargs):
    """Return a dict where missing keys are filled in by defaults

    >>> options = dict(hello=0)
    >>> with_defaults(options, hello=4, world=5) == dict(hello=0, world=5)
    True
    >>> defaults = dict(hello=4, world=5)
    >>> with_defaults(options, defaults) == dict(hello=0, world=5)
    True
    >>> with_defaults(options, defaults, world=7, yes=3) == dict(hello=0, world=5, yes=3)
    True
    """
    options = options if options else {}
    if defaults_dict:
        options = dict(defaults_dict, **options)
    return dict(defaults_kwargs, **options)


def x_pi(value):
    """Return str of value in 'multiples of pi' latex representation

    >>> x_pi(6.28) == r"$2\pi$"
    True
    >>> x_pi(3) == r"$0.95\pi$"
    True
    >>> x_pi(-np.pi) == r"$-\pi$"
    True
    >>> x_pi(0) == "0"
    True
    """
    n = value / np.pi
    if np.isclose(n, 0):
        return "0"
    elif np.isclose(abs(n), 1):
        return r"$\pi$" if n > 0 else r"$-\pi$"
    else:
        return r"${:.2g}\pi$".format(n)