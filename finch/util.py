import copy
import dataclasses
import functools
import inspect
import numbers
import os
import pathlib
import re
import shutil
import socket
import types
import typing
from collections.abc import Callable
from contextlib import closing
from typing import Any, Literal, Type, TypeGuard, TypeVar

import tqdm
from wonderwords import RandomWord  # type: ignore

from ._util import arg2list, inverse, map_keys, parse_bool  # noqa: F401

###############################
# Networking utilities
###############################


def check_socket_open(host: str = "localhost", port: int = 80) -> bool:
    """
    Return whether a port is in use / open (``True``) or not (``False``).

    Args:
        host (str): The hostname
        port (int): The port to check

    Returns:
        bool: Whether the given port at the given host is open or not.

    Group:
        Util
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0


###############################
# Path utilities
###############################


PathLike = os.PathLike | str
"""
Type alias for path types, as recommended py PEP 519.

Group:
    Util
"""


def get_absolute(path: PathLike, context: PathLike = ".") -> pathlib.Path:
    """
    Return the abolute path in the given context if a relative path was given.
    If an absolute path is given, it is directly returned.

    Args:
        path (PathLike): The absolute or relative path.
        context (PathLike, optional): The context for a relative path.
            Defaults to the current working directory.

    Group:
        Util
    """
    path = pathlib.Path(path)
    context = pathlib.Path(context).absolute()
    if not path.is_absolute():
        path = pathlib.Path(context, path)
    return path


def get_path(*args: PathLike) -> pathlib.Path:
    """
    Returns a new path by joining the given path arguments.
    If the directories do not exist yet, they will be created.

    Group:
        Util
    """
    out = pathlib.Path(*args)
    to_make = out if out.suffix == "" else out.parent
    to_make.mkdir(parents=True, exist_ok=True)
    return out


def remove_if_exists(path: PathLike) -> pathlib.Path:
    """
    Removes the given directory if it exists and returns the original path.

    Group:
        Util
    """
    path = pathlib.Path(path)
    if path.exists():
        shutil.rmtree(path)
    return path


def clear_dir(path: PathLike) -> None:
    """
    Removes the content of the given directory.

    Group:
        Util
    """
    path = pathlib.Path(path)
    if not path.is_dir():
        raise ValueError("Given path is not a directory", path)
    for file in path.iterdir():
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)


###############################
# Progres bar utilities
###############################


PbarArg = bool | tqdm.tqdm
"""
Argument type for handling progress bars.
Functions accepting the progress bar argument support outputting their progress via a tqdm progress bar.
The argument can then either be a boolean,
indicating that a new progress bar should be created, or no progress bar should be used at all,
or it can be a preexisting progress bar which will be updated.

Group:
    Util
"""


def get_pbar(pbar: PbarArg, iterations: int) -> tqdm.tqdm | None:
    """
    Convenience function for handling progress bar arguments.
    This makes sure that a ``tqdm`` progress bar is returned if one is requested, or ``None``.

    Args:
        pbar (PbarArg): The progress bar argument
        iterations (int): The number of iterations to perform if a new progress bar was requested.

    Returns:
        A ``tqdm`` progress bar, or None if no progress bar was requested.

    See Also:
        :py:const:`finch.util.PbarArg`

    Group:
        Util
    """
    if isinstance(pbar, bool):
        if pbar:
            return tqdm.trange(iterations)
        else:
            return None
    else:
        return pbar


###############################
# Reflection utilities
###############################

T = TypeVar("T")


def fill_none_properties(x: T, y: T) -> T:
    """
    Return ``x`` as a copy, where every attribute which is ``None`` is set to the attribute of ``y``.

    Group:
        Util
    """
    out = copy.copy(x)
    to_update = {k: y.__dict__[k] for k in x.__dict__ if x.__dict__[k] is None}
    out.__dict__.update(to_update)
    return out


def add_missing_properties(x: T, y: object) -> T:
    """
    Return ``x`` as a copy, with attributes from ``y`` added to ``x`` which were not already present.

    Group:
        Util
    """
    out = copy.copy(x)
    to_update = {k: y.__dict__[k] for k in y.__dict__ if k not in x.__dict__}
    out.__dict__.update(to_update)
    return out


def equals_not_none(x: object, y: object) -> bool:
    """
    Compares the common properties of the two given objects.
    Return ``True`` if the not-None properties present in both object are all equal.

    Group:
        Util
    """
    xd = x.__dict__
    yd = y.__dict__
    vs = set(xd.keys()).intersection(yd.keys())
    return all(xd[v] is not None and yd[v] is not None and xd[v] != yd[v] for v in vs)


def has_attributes(x: object, y: object, excludes: typing.Iterable[str] = []) -> bool:
    """
    Return true if y has the same not-None attributes as x.

    Args:
        excludes (Iterable[str], optional):
            Attributes to exclude from the comparison.
            Defaults to no exclusions.

    Group:
        Util
    """
    xd = x.__dict__
    yd = y.__dict__
    excl_set = set(excludes)
    return all(xd[v] is None or (v in yd and xd[v] == yd[v]) for v in xd if v not in excl_set)


def get_class_attribute_names(cls: type, excludes: list[str] = []) -> list[str]:
    """
    Return the attribute names of a class.

    Args:
        cls (type): The class to extract the attribute names from
        exlcudes (list[str]): A list of attribute names to exclude from the returned list.

    Group:
        Util
    """
    dummy = dir(type("dummy", (object,), {}))  # create a new dummy class and extract its attributes
    attr = inspect.getmembers(cls, lambda x: not inspect.isroutine(x))
    str_attr = [a[0] for a in attr]
    return [
        a for a in str_attr if a not in dummy and not (a.startswith("__") and a.endswith("__")) and a not in excludes
    ]


def get_class_attributes(obj: object) -> dict[str, Any]:
    """Return the class attributes of an object as a dictionary.

    Group:
        Util
    """
    names = set(get_class_attribute_names(obj.__class__))
    return {k: v for k, v in obj.__dict__.items() if k in names}


def sig_matches_hint(sig: inspect.Signature, hint: Any) -> bool:  # TODO: Adjust type of hint, if possible
    """
    Return ``True`` if the function signature and the ``Callable`` type hint match.

    Args:
        sig (inspect.Signature): The function signature
        hint (Any): The type hint

    Returns:
        bool: ``False`` if the type hint is either not ``Callable``, if the arguments don't match or the
            return type doesn't match. ``True`` otherwise.

    Group:
        Util
    """
    if typing.get_origin(hint) != Callable:
        return False
    params, ret = typing.get_args(hint)

    if sig.return_annotation != ret:
        return False

    required_params = [ph for ph in sig.parameters.values() if ph.default is ph.empty]
    default_params = [ph for ph in sig.parameters.values() if ph.default is not ph.empty]

    if len(required_params) > len(params) or len(params) > len(sig.parameters):
        return False

    required_match = all(ps.annotation == ph for ps, ph in zip(required_params, params[: len(required_params)]))
    if not required_match:
        return False

    remaining = params[len(required_params) :]  # noqa: E203
    j = 0
    for r in remaining:
        while default_params[j].annotation != r and j < len(default_params):
            j += 1
        if j == len(default_params):
            return False
        j += 1
    return True


def list_funcs_matching(
    module: types.ModuleType, regex: str | None = None, type_hint: Any = None
) -> list[Callable]:  # TODO: adjust type of type_hint if possible
    """
    Returns a list of functions from a module matching the given parameters.

    Args
        module (ModuleType): The module from which to list the functions
        regex (str, optional): A regex which matches the function names to be returned.
            Defaults to None, which means that no regex should be checked.
        signature (Any, optional):
            A ``Callable`` type hint specifying the signature of the functions to be returned.
            Defaults to None, which means that the function signature won't be checked.

    Group:
        Util
    """
    out = [
        f
        for name, f in inspect.getmembers(module, inspect.isfunction)
        if (regex is None or re.match(regex, name))
        and (type_hint is None or sig_matches_hint(inspect.signature(f), type_hint))
    ]
    return out


def get_primitive_attrs_from_dataclass(dc: object) -> dict[str, str | numbers.Number]:
    """
    Returns the flattened fields from a dataclass as primitives.
    """
    assert dataclasses.is_dataclass(dc)
    attrs = flatten_dict(dataclasses.asdict(dc))
    # transform non-numeric attributes to strings
    out = dict()
    for k, v in attrs.items():
        if isinstance(v, Callable):  # type: ignore
            # extract function name from callable
            if isinstance(v, functools.partial):
                v_str = v.func.__name__
                if len(v.args) > 0:
                    v_str += "_" + "_".join(str(a) for a in v.args)
                if len(v.keywords) > 0:
                    v_str += "_" + "_".join(k + "=" + str(v) for k, v in v.keywords.items())
                v = v_str
            else:
                v = v.__name__
        elif dataclasses.is_dataclass(v):
            v = get_primitive_attrs_from_dataclass(v)  # recursive
        elif not isinstance(v, numbers.Number):
            v = str(v)
        elif isinstance(v, bool):
            v = str(v)
        out[k] = v
    out = flatten_dict(out)  # dataclasses were transformed to dicts. So flatten them.
    return out


###############################
# Config class
###############################


class Config:
    """
    Base class for configuration classes.
    Classes inheriting from this class must be dataclasses (with the @dataclass decorator).

    Group:
        Util
    """

    @classmethod
    def list_configs(cls, **kwargs: Any) -> list:
        """
        Returns a list of run configurations,
        which is the euclidean product between the given lists of individual configurations.
        """
        configs: list[dict[str, Any]] = []
        for arg, vals in kwargs.items():
            if not isinstance(vals, list):
                vals = [vals]
            updates = [{arg: v} for v in vals]
            if len(configs) == 0:
                configs = updates
            else:
                configs = [c | u for c in configs for u in updates]
        return [cls(**c) for c in configs]


###############################
# Datastructure utilities
###############################


def flatten_dict(d: dict, separator: str = "_") -> dict:
    """
    Flattens a dictionary. The keys of the inner dictionary are appended to the outer key with the given separator.

    Group:
        Util
    """
    out = dict()
    flat = True
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[k + separator + kk] = vv
                if isinstance(vv, dict):
                    flat = False
        else:
            out[k] = v
    if not flat:
        return flatten_dict(out, separator)
    else:
        return out


def flat_list(arg: Any) -> list:
    """Creates a flat list from the argument.
    The argument can be any object.
    If it is not a list, a single-element list is returned.
    If it is a list, a flattened version of this list will be returned.

    Args:
        arg: The argument to flatten

    Returns:
        list: The flattened argument
    """
    if isinstance(arg, list):
        return [aa for a in arg for aa in flat_list(a)]
    else:
        return [arg]


def recursive_update(d: dict, updates: dict) -> dict:
    """Returns a copy of ``d`` with its content replaced by ``updates`` wherever specified.
    Nested dictionaries won't be replaced, but updated recursively as specified by ``updates``.

    Args:
        d (dict): The dictionary to update
        updates (dict): The updates to perform recursively

    Returns:
        dict: The updated dictionary

    Group:
        Util
    """
    out = dict()
    for k, v in d.items():
        if k not in updates:
            out[k] = v
        elif isinstance(v, dict) and isinstance(updates[k], dict):
            out[k] = recursive_update(v, updates[k])
        else:
            out[k] = updates[k]
    for k, v in updates.items():
        if k not in d:
            out[k] = v
    return out


class RecursiveNamespace(types.SimpleNamespace):
    """
    A ``types.SimpleNamespace`` which can handle nested dictionaries.

    Group:
        Util
    """

    @staticmethod
    def map_entry(entry: T) -> T | "RecursiveNamespace":
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, RecursiveNamespace(**val))
            elif isinstance(val, list):
                setattr(self, key, list(map(self.map_entry, val)))


###############################
# Typing utilities
###############################


def is_list_of(val: list[Any], typ: Type[T]) -> TypeGuard[list[T]]:
    """
    Type guard for checking lists.

    Group:
        Util
    """
    return all(isinstance(v, typ) for v in val)


def is_2d_list_of(val: list[list[Any]], typ: Type[T]) -> TypeGuard[list[list[T]]]:
    """Type guard for checking lists of lists.

    Group:
        Util
    """
    return all(all(isinstance(vv, typ) for vv in v) for v in val)


def is_callable_list(val: list[Any]) -> TypeGuard[list[Callable]]:
    """Type guard for checking that a list contains callable objects.

    Group:
        Util
    """
    return all(callable(v) for v in val)


###############################
# Miscellaneous utilities
###############################


def random_entity_name(excludes: list[str] = []) -> str:
    """
    Return a random name for an entity, such as a file or a variable.

    Args:
        excludes (list[str], optional):
            A list of words to exclude from generation.
            Defaults to an empty list.

    Group:
        Util
    """
    excl = set(excludes)
    r = RandomWord()
    out = None
    while out is None or out in excl:
        adj = r.word(include_parts_of_speech=["adjectives"])
        noun = r.word(include_parts_of_speech=["nouns"])
        out = adj + "_" + noun
    return out


def funcs_from_args(f: Callable, args: list[dict]) -> list[Callable]:
    """
    Takes a function `f` and a list of arguments `args` and
    returns a list of functions which are the partial applications of `f` onto `args`.

    Group:
        Util
    """
    return [functools.partial(f, **a) for a in args]


ImgSuffix = Literal["eps", "jpeg", "jpg", "pdf", "pgf", "png", "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff"]
"""
A literal for image file suffixes.

Group:
    Util
"""
