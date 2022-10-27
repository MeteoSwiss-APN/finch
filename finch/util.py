from ast import Call, arg
from contextlib import closing
from dataclasses import dataclass, is_dataclass
import dataclasses
import functools
import numbers
import pathlib
import socket
import types
from typing import Dict, List, TypeVar, Any
from collections.abc import Callable
import typing
import dask.array as da
import xarray as xr
import numpy as np
import inspect
import re
from dask_jobqueue.slurm import SLURMJob
from . import environment as env
import tqdm
from wonderwords import RandomWord
import copy

def adjust_dims(dims: List[str], array: xr.DataArray) -> xr.DataArray:
    """
    Returns a new DataArray with the same content as `array` such that the dimensions match `dims` in content and order.
    This is achieved with a combination of `expand_dims`, `squeeze` and `transform`.
    When trying to remove dimensions with sizes larger than 1, an error will be thrown.
    """
    # Performance might be improved. We could check whether the dimensions in the array are in correct order.
    # If so, we wouldn't need a transpose. Since we are however working with views, this might not even be faster.
    to_remove = set(array.dims).difference(dims)
    to_add = set(dims).difference(array.dims)
    return array.squeeze(to_remove).expand_dims(list(to_add)).transpose(*dims)

def custom_map_blocks(f: Callable, *args: xr.DataArray | xr.Dataset, 
    name: str = None, dtype = None, template: xr.DataArray = None, f_in_out_type = np.ndarray, 
    **kwargs) -> xr.DataArray:
    """
    Custom implementation of map_blocks from dask for xarray data arrays based on dask's `blockwise` and `map_blocks` functions.

    Arguments:
    ---
    - f: Callable. The function to be executed on the chunks of `args`.
    The input / output of the chunks can be controlled with the `f_in_out_type` argument.
    - args: xarray.DataArray | xarray.Dataset. The data array arguments to `f`.
        If a dataset is passed, its non-coordinate variables will be extracted and used as inputs to `f`.
    The first element will be used as a template for the output where 
    - name: str, optional. The name of the output array. Defaults to the name of `f`.
    - dtype: type, optional. The type of the elements of the output array. Defaults to the dtype of `template`.
    - template: xr.DataArray, optional. A template array used to determine some characteristics of the output array.
    The coordinates, attributes and the dimensions will be copied from the template into the output array.
    Additionally the template's data will be used to determine the type (e.g. dask array) and the dtype (e.g. np.float64) of the data of the output.
    The content of the data is irrelevant (the array can be empty).
    Defaults to using the first element of `args` as a template argument.
    - f_in_out_type: optional. The type of the input arrays for `f`.
    Can be either `numpy.ndarray`, `dask.array.Array` or `xarray.DataArray`, providing a performance penalty increasing in this order.
    Note that this parameter only serves as a convenience to support the different interfaces for data handling.
    The data chunks must still fully fit into memory.
    Defaults to `numpy.ndarray`.
    """
    # extract dataset
    if isinstance(args[0], xr.Dataset):
        assert len(args) == 1, "Only one dataset can be passed."
        dataset = args[0]
        args = list(dataset.data_vars.values())

    # handle optional arguments
    if template is None:
        template: xr.DataArray = args[0].copy()
    if dtype is None:
        dtype = template.dtype
    if name is None:
        name = f.__name__
    # identify the individual dimensions while keeping the order in which they occur
    dims = list(dict.fromkeys([d for a in args for d in a.dims]))

    if f_in_out_type is xr.DataArray:
        # we need to use map_blocks here since it supports the block_info argument, 
        # which we use to construct coordinates for the chunks
        # map_blocks expects all arguments to have the same number of dimensions
        xr_args = [adjust_dims(dims, a).data for a in args]
        # a helper function for retrieving the coordinates of a data array chunk
        def get_chunk_coords(array_location: List[slice], array: xr.DataArray) -> Dict[str, xr.DataArray]:
            dim_ind_map = {d:i for d, i in zip(array.dims, range(len(array.dims)))}
            coord_dims = set(array.dims).intersection(array.coords.keys())
            # add non-index coordinates
            out = {d : array.coords[d] for d in array.coords.keys() if d not in coord_dims}
            # add index coordinates
            for d in coord_dims:
                s = array_location[dim_ind_map[d]]
                c = array.coords[d][s]
                out[d] = c
            return out
        # This function wraps `f` such that it can accept numpy arrays.
        # It creates xr.DataArrays from the numpy arrays with the appropriate metadata before calling `f`.
        # Afterwards, the underlying numpy array is extracted from the data array.
        def xr_wrap(*chunks: np.ndarray, block_info, **kwargs):
            xr_chunks = [
                    adjust_dims( # readjust dimensions of the chunk according to the dimensions of the full array
                        a.dims,
                        xr.DataArray(c, coords=get_chunk_coords(info["array-location"], a), dims=dims, attrs=a.attrs)
                    )
                    for c, info, a in zip(chunks, block_info.values(), args)
            ]
            return np.array(f(*xr_chunks, **kwargs).data)
        # run map_blocks
        out = da.map_blocks(xr_wrap, *xr_args, name=name, dtype=dtype, meta=template.data, **kwargs)
    else:
        # we can use da.blockwise for dask and numpy arrays, which reduces some overhead compared to map_blocks
        # da.blockwise requires array-index pairs, which we can easily generate from the dimension names
        index_map = {k:v for k, v in zip(dims, range(len(dims)))}
        index = [tuple([index_map[d] for d in a.dims]) for a in args]
        out_ind = index[0]
        da_args = [a.data for a in args]
        block_args = [x for z in zip(da_args, index) for x in z]

        if f_in_out_type is da.Array:
            # wrap `f` for numpy array in- and output
            f = lambda *chunks, **kwargs: np.array(f(*[da.Array(c) for c in chunks], **kwargs))
        # run da.blockwise
        out = da.blockwise(f, out_ind, *block_args, name=name, dtype=dtype, meta=template.data, **kwargs)
    # out is now a dask array, which should be converted to an xarray data array
    return xr.DataArray(out, name=name, attrs=template.attrs, coords=template.coords, dims=template.dims)

def check_socket_open(host: str = "localhost", port: int = 80) -> bool:
    """Returns whether a port is in use / open (`True`) or not (`False`)."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0

def sig_matches_hint(
    sig: inspect.Signature,
    hint: type
) -> bool:
    """
    Returns `True` if the function signature and the `Callable` type hint match.
    """
    if typing.get_origin(hint) != Callable:
        return False
    params, ret = typing.get_args(hint)
    if sig.return_annotation != ret:
        return False
    if len(sig.parameters) != len(params):
        return False
    return all(
        ps.annotation == ph for ps, ph in zip(sig.parameters.values(), params)
    )
    

def list_funcs_matching(module: types.ModuleType, regex: str | None = None, type_hint: type | None = None) -> List[Callable]:
    """
    Returns a list of functions from a module matching the given parameters.

    Arguments:
    ---
    - module: ModuleType. The module from which to list the functions
    - regex: str, optional. A regex which matches the function names to be returned.
    - signature: type, optional. A `Callable` type hint specifying the signature of the functions to be returned.
    """
    out = [
        f 
        for name, f in inspect.getmembers(module, inspect.isfunction)
        if (regex is None or re.match(regex, name)) and \
            (type_hint is None or sig_matches_hint(inspect.signature(f), type_hint))
    ]
    return out

class SLURMRunner(SLURMJob):
    """
    Instances of this class can execute arbitrary shell commands on the slurm cluster.
    """
    def __init__(self, name=None, queue=None, project=None, account=None, walltime=None, job_cpu=None, job_mem=None, config_name=None):
        super().__init__(None, name, queue, project, account, walltime, job_cpu, job_mem, None)

    async def start(self, cmd: list[str]):
        self._command_template = " ".join(map(str, cmd))
        await super().start()

def get_absolute(path: pathlib.Path | str, context: pathlib.Path | str = env.proj_root) -> pathlib.Path | str:
    """
    Returns the abolute path in the given context if a relative path was given.
    If an absolute path is given, it is directly returned.
    Both `pathlib.Path` and `str` repesentations are accepted. The return type depends on which type `path` has.
    """
    ispathlib = isinstance(path, pathlib.Path)
    path = pathlib.Path(path)
    context = pathlib.Path(context).absolute()
    if not path.is_absolute():
        path = pathlib.Path(context, path)
    if ispathlib:
        return path
    else:
        return str(path)

def funcs_from_args(f: Callable, args: list[dict]) -> list[Callable]:
    """
    Takes a function `f` and a list of arguments `args` and 
    returns a list of functions which are the partial applications of `f` onto `args`.
    """
    return [functools.partial(f, **a) for a in args]

PbarArg = bool | tqdm.tqdm
"""
Argument type for handling progress bars.
Functions accepting the progress bar argument support outputting their progress via a tqdm progress bar.
The argument can then either be a boolean, indicating that a new progress bar should be created, or no progress bar should be used at all,
or it can be a preexisting progress bar which will be updated.
"""

def get_pbar(pbar: PbarArg, iterations: int) -> tqdm.tqdm:
    """
    Convenience function for progress bar arguments.
    This makes sure that a `tqdm` progress bar is returned if one is requested, or `None`.
    """
    if isinstance(pbar, bool):
        if pbar:
            return tqdm.trange(iterations)
        else:
            return None
    else:
        return pbar

def random_entity_name(excludes: list[str] = []) -> str:
    """
    Returns a random name for an entity, such as a file or a variable.
    """
    excludes = set(excludes)
    r = RandomWord()
    out = None
    while out is None or out in excludes:
        adj = r.word(include_parts_of_speech=["adjectives"])
        noun = r.word(include_parts_of_speech=["nouns"])
        out = adj + "_" + noun
    return out

T = TypeVar("T")
def fill_none_properties(x: T, y: T) -> T:
    """
    Returns `x` as a copy, where every attribute which is `None` is set to the attribute of `y`.
    """
    out = copy.copy(x)
    to_update = {k: y.__dict__[k] for k in x.__dict__ if x.__dict__[k] is None}
    out.__dict__.update(to_update)
    return out

T = TypeVar("T")
def add_missing_properties(x: T, y) -> T:
    """
    Returns `x` as a copy, with attributes from `y` added to `x` which were not already present.
    """
    out = copy.copy(x)
    to_update = {k: y.__dict[k] for k in y.__dict__ if k not in x.__dict__}
    out.__dict__.update(to_update)
    return out

def equals_not_none(x, y) -> bool:
    """
    Returns true if the two given objects are equal on all properties except for those for which one of them is `None` or not defined.
    """
    xd = x.__dict__
    yd = y.__dict__
    vs = set(xd.keys()).intersection(yd.keys())
    return all(
        xd[v] is not None and yd[v] is not None and xd[v] != yd[v]
        for v in vs
    )

def has_attributes(x, y) -> bool:
    """
    Returns true if y has the same not-None attributes as x.
    """
    xd = x.__dict__
    yd = y.__dict__
    return all(
        xd[v] is None or (v in yd and xd[v] == yd[v])
        for v in xd
    )

def get_class_attributes(cls: type, excludes: list[str] = []) -> list[str]:
    """
    Returns the attributes of a class.
    """
    dummy = dir(type("dummy", (object,), {})) # create a new dummy class and extract its attributes
    attr = inspect.getmembers(cls, lambda x: not inspect.isroutine(x))
    attr = [a[0] for a in attr]
    return [
        a for a in attr if 
        a not in dummy 
        and not (a.startswith("__") and a.endswith("__"))
        and not a in excludes
    ]

def flatten_dict(d: dict, separator: str = "_") -> dict:
    """
    Flattens a dictionary. The keys of the inner dictionary are appended to the outer key with the given separator.
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

class Config():
    """
    Base class for configuration types. 
    Classes inheriting from this class must be dataclasses (with the @dataclass decorator).
    """
    @classmethod
    def list_configs(cls, **kwargs) -> list:
        """
        Returns a list of run configurations, which is the euclidean product between the given lists of individual configurations.
        """
        configs: list[dict[str, Any]] = []
        for arg in kwargs:
            vals = kwargs[arg]
            if not isinstance(vals, list):
                vals = [vals]
            updates = [{arg : v} for v in vals]
            if len(configs) == 0:
                configs = updates
            else:
                configs = [c | u for c in configs for u in updates]
        return [cls(**c) for c in configs]


def get_primitive_attrs_from_dataclass(dc) -> dict[str, str | numbers.Number]:
    """
    Returns the flattened fields from a dataclass as primitives.
    """
    assert dataclasses.is_dataclass(dc)
    attrs = flatten_dict(dataclasses.asdict(dc))
    # transform non-numeric attributes to strings
    out = dict()
    for k, v in attrs.items():
        if isinstance(v, Callable): # extract function name from callable
            if isinstance(v, functools.partial):
                v_str = v.func.__name__
                v_str += "_" + "_".join(str(a) for a in v.args)
                v_str += "_" + "_".join(k + "=" + str(v) for k, v in v.keywords.items())
                v = v_str
            else:
                v = v.__name__
        elif dataclasses.is_dataclass(v):
            v = get_primitive_attrs_from_dataclass(v) # recursive
        elif not isinstance(v, numbers.Number):
            v = str(v)
        out[k] = v
    out = flatten_dict(out) # dataclasses were transformed to dicts. So flatten them.
    return out

def parse_bool(b: str) -> bool:
    b = b.lower()
    if b == "true":
        return True
    elif b == "false":
        return False
    else:
        raise ValueError(f"Could not parse {b} to boolean type.")
