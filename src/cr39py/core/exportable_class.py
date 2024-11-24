import datetime
import importlib
import inspect
from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from cr39py.core.units import unit_registry as u


class ExportableClassMixin:
    """
    A class that can be exported to an HDF5 file
    """

    # List of attributes to write to HDF5 file
    _exportable_attributes = []

    def to_hdf5(self, path: Path | str):
        path = Path(path)
        tmp_path = Path(path.parent, path.stem + "_tmp" + path.suffix)
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.unlink(missing_ok=True)

        with h5py.File(tmp_path, "a") as f:
            self._write_class_to_hdf5(f, self)

        # If successful to this point, delete the existing file and rename
        # the temp file to replace it
        path.unlink(missing_ok=True)
        tmp_path.rename(path)

    def _write_class_to_hdf5(self, group: h5py.Group, obj):

        # Empty the group before saving new data there
        for key in group.keys():
            del group[key]

        group.attrs["type"] = "ExportableClassMixin"
        # Save the module location and the name of the class
        # which will be used to retrieve it
        group.attrs["class_module"] = str(obj.__class__.__module__)
        group.attrs["class_name"] = str(obj.__class__.__name__)
        group.attrs["timestamp"] = datetime.datetime.now().isoformat()

        for name in self._exportable_attributes:
            if hasattr(self, name):
                self._write_hdf5_entry(group, name, getattr(self, name))

    def _write_hdf5_entry(self, group: h5py.Group, name: str, obj: object, attrs={}):
        """
        Writes values to HDF5
        """

        if isinstance(obj, ExportableClassMixin):
            grp = group.require_group(name)
            obj._write_class_to_hdf5(grp, obj)

            group[name].attrs["type"] = "ExportableClassMixin"
            # Save the module location and the name of the class
            # which will be used to retrieve it
            group[name].attrs["class_module"] = str(obj.__class__.__module__)
            group[name].attrs["class_name"] = str(obj.__class__.__name__)

        elif obj is None:
            group[name] = "None"
            group[name].attrs["type"] = "None"

        elif isinstance(obj, (bool, float, int)):
            group[name] = obj
            group[name].attrs["type"] = str(obj.__class__.__name__)

        elif isinstance(obj, str):
            group[name] = obj
            group[name].attrs["type"] = "str"

        elif type(obj).__module__ == "numpy":
            if obj.size > 10:
                compression = "gzip"
                compression_opts = 5
            else:
                compression = None
                compression_opts = None

            group.create_dataset(
                name,
                data=obj,
                compression=compression,
                compression_opts=compression_opts,
            )

            # Store the numpy dtype in the type field
            group[name].attrs["type"] = f"numpy.{type(obj).__name__}"

        elif isinstance(obj, u.Quantity):
            group[name] = obj.m
            group[name].attrs["unit"] = str(obj.u)
            group[name].attrs["type"] = "pint.Quantity"

        elif isinstance(obj, (list, tuple)):
            list_group = group.require_group(name)
            list_group.attrs["type"] = type(obj).__name__

            for i, item in enumerate(obj):
                self._write_hdf5_entry(list_group, f"item{i}", item)

        elif isinstance(obj, dict):
            dict_group = group.require_group(name)
            dict_group.attrs["type"] = "dict"

            for i, key in enumerate(obj.keys()):
                # Write the key and value separately
                # Necessary because not all valid python dict keys
                # are valid h5 names (strings)
                entry_group = dict_group.require_group(f"item{i}")
                self._write_hdf5_entry(entry_group, "key", key)
                self._write_hdf5_entry(entry_group, "value", obj[key])

        else:
            raise NotImplementedError(f"Saving object type {type(obj)} not supported.")

        for key, val in attrs.items():
            group[name].attrs[key] = val

    @classmethod
    def from_hdf5(cls, path: Path | str, group: str | None = None):
        path = Path(path)

        with h5py.File(path, "r") as f:

            if group is None:
                grp = f
            else:
                grp = f[group]

            objtype = grp.attrs["class_name"]

            if objtype == cls.__name__:
                obj = cls()._read_class_from_hdf5(grp, classref=cls)
            else:
                raise ValueError(
                    f"Type of object {objtype} does not match "
                    "the type of this object "
                    f"{cls.__name__}"
                )

        return obj

    def _read_class_from_hdf5(self, group: h5py.Group, classref=None):
        """
        Reads and reinstantiates a class represented by a group in an
        hdf5 file.
        """

        # If a class reference is passed, assume the class is of this type
        # this saves a lookup from the `from_hdf5` method.
        if classref is None:
            # Create an object of the appropriate class
            module = importlib.import_module(group.attrs["class_module"])
            _class = getattr(module, group.attrs["class_name"])
        else:
            _class = classref
        obj = _class()

        for name, entry in group.items():
            val = obj._read_hdf5_entry(entry)

            # Try to set the attribute
            # If it doesn't work, it's probably a property in which case
            # we will skip setting it
            try:
                setattr(obj, name, val)
            except AttributeError:
                pass

        return obj

    def _read_hdf5_entry(self, entry: [h5py.Group, h5py.Dataset]):
        """
        Reads and returns entry from an hdf5 file.
        """
        dtype = entry.attrs["type"]

        if dtype == "ExportableClassMixin":
            return self._read_class_from_hdf5(entry)

        elif dtype == "None":
            return None

        elif dtype == "int":
            return int(entry[...])

        elif dtype == "float":
            return float(entry[...])

        elif dtype == "bool":
            return bool(entry[...])

        elif dtype == "str":
            return entry[()].decode("utf-8")

        elif dtype == "list":
            return [self._read_hdf5_entry(entry[key]) for key in entry.keys()]

        elif dtype == "tuple":
            return tuple([self._read_hdf5_entry(entry[key]) for key in entry.keys()])

        elif dtype == "dict":
            out = {}
            for name, grp in entry.items():
                key = self._read_hdf5_entry(grp["key"])
                value = self._read_hdf5_entry(grp["value"])
                out[key] = value
            return out

        elif "numpy" in dtype:
            # Extract the numy dtype substring from the type attribute
            np_dtype = dtype[6:]
            if np_dtype == "ndarray":
                return entry[...]
            else:
                return entry[...].astype(np_dtype)

        elif dtype == "pint.Quantity":
            return u.Quantity(entry[...], entry.attrs["unit"])

        else:
            raise ValueError(f"Unrecognized dtype {dtype} for entry {entry}")


def saveable_class(attributes: list[str] | None = None) -> Callable:
    """
    The saveable class decorator attaches all exportable attributes of parent
    classes to the decorated class.

    Also, you can attach attributes this way if you want.
    """

    if attributes is None:
        attributes = []

    def decorator(cls):

        if not issubclass(cls, ExportableClassMixin):
            raise ValueError(
                "Only subclasses of ExportableClassMixin can be decorated with"
                "saveable_class"
            )

        exportable_attributes = []

        # Add attributes from parent classes
        for parent in reversed(inspect.getmro(cls)):
            if hasattr(parent, "_exportable_attributes"):
                for attr in parent._exportable_attributes:
                    res = attr.split(".")

                    symbol = res[1] if len(res) > 1 else res[0]
                    exportable_attributes.append(symbol)

        for attr in attributes:
            exportable_attributes.append(attr)

        cls._exportable_attributes = list(set(exportable_attributes))

        return cls

    return decorator
