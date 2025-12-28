"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import numpy as np
from enum import Enum

import base64
import binascii
import json as json_
import ujson as json

from .logging import Printer


kVerbose = False

if not kVerbose:

    def print(*args, **kwargs):
        pass


# Global registry dictionary used for registering the serializable classes below
class_registry = {}


def register_class(cls):
    """
    Decorator to register a class in the registry.
    """
    class_registry[cls.__name__] = cls
    return cls


# NOTE: each child class must use @register_class to get registered in the class registry
@register_class
class SerializableEnum(Enum):
    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

    def to_json(self):
        # Serialize the full class name and enum name
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def from_json(cls, serialized_str):
        # Deserialize by extracting the class name and enum name
        try:
            class_name, enum_name = serialized_str.split(".")
            # Dynamically get the enum class from the module
            enum_class = cls.get_enum_class(class_name)
            return enum_class[enum_name]
        except (ValueError, KeyError) as e:
            raise ValueError(
                f"Invalid serialized string '{serialized_str}' for enum '{cls.__name__}'"
            ) from e

    @staticmethod
    def get_enum_class(class_name):
        # Look for the class in the registry first
        if class_name in class_registry:
            return class_registry[class_name]

        # Otherwise, look for the class among subclasses
        for subclass in SerializableEnum.__subclasses__():
            if subclass.__name__ == class_name:
                return subclass
        raise ValueError(f"Enum class '{class_name}' not found.")

    @classmethod
    def is_member(cls, name):
        return name in cls.__members__

    @classmethod
    def deserialize_if_member(cls, value):
        print(f"SerializableEnum.deserialize_if_member: value {value}")
        if isinstance(value, str):
            try:
                # Try to deserialize using from_json without checking if it's a member first
                ret = cls.from_json(value)
                print(f"\tSerializableEnum.deserialize_if_member: got ret {ret}")
                return ret
            except ValueError:
                return value
        return value


def format_floats_for_json(obj):
    """
    Recursively format floats in a data structure to ensure consistent JSON representation.
    This helps avoid floating point differences when saving and reloading maps.

    Uses a consistent formatting approach:
    - For regular floats: uses repr() which gives the shortest representation that round-trips correctly
    - This ensures the same float value always produces the same string representation
    - Preserves full precision for IEEE 754 double precision (53 bits of precision)
    """
    if isinstance(obj, dict):
        return {k: format_floats_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_floats_for_json(item) for item in obj]
    elif isinstance(obj, float):
        # Use repr() which gives the shortest representation that round-trips correctly
        # This ensures consistent formatting: the same float value always produces the same string
        # Example: 1.0 -> "1.0", 0.1 -> "0.1", 1.2345678901234567 -> "1.2345678901234567"
        # This is better than fixed precision which might add unnecessary zeros
        return float(repr(obj))
    elif isinstance(obj, np.floating):
        # Handle numpy float types (float32, float64, etc.)
        # Convert to Python float first, then use repr() for consistent formatting
        return float(repr(float(obj)))
    elif isinstance(obj, np.integer):
        # Handle numpy integer types
        return int(obj)
    elif isinstance(obj, np.ndarray):
        # For numpy arrays, convert to list and format recursively
        return format_floats_for_json(obj.tolist())
    return obj


class PreciseFloatEncoder(json_.JSONEncoder):
    """
    Custom JSON encoder that ensures consistent float formatting with maximum precision.
    This helps avoid floating point differences when saving and reloading maps.

    Note: Works with standard json module. For ujson, use format_floats_for_json()
    to pre-process the data before serialization.
    """

    def encode(self, obj):
        # Recursively process the object to format floats consistently
        formatted_obj = format_floats_for_json(obj)
        return super().encode(formatted_obj)

    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return format_floats_for_json(obj.tolist())
        elif isinstance(obj, np.floating):
            return float(repr(float(obj)))
        elif isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class SerializableEnumEncoder(PreciseFloatEncoder):
    def default(self, obj):
        if isinstance(obj, SerializableEnum):
            return obj.to_json()
        if isinstance(obj, Serializable):
            return json.loads(obj.to_json())
        return super().default(obj)


# Define Serializable for regular objects with nested structures
# NOTE: each child class must use @register_class to get registered in the class registry
@register_class
class Serializable:
    def to_json(self):
        """
        Convert the object to JSON format (string).
        This method will serialize the object's dictionary.
        """
        dict_data = self.serialize(self.__dict__)
        dict_data["class_name"] = (
            self.__class__.__name__
        )  # add this information to get it recognizable
        return json.dumps(dict_data)

    @classmethod
    def from_json(cls, json_str):
        """
        Deserialize a JSON string back into an object.
        This method will convert the JSON string to a dictionary
        and use it to initialize an instance of the class.
        """
        print
        data = json.loads(json_str)
        return cls.deserialize(data)

    @classmethod
    def serialize(cls, data):
        """
        Recursively serialize data, converting instances of Serializable
        subclasses to their JSON representation (to_dict).
        """
        if isinstance(data, dict):
            return {k: cls.serialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.serialize(v) for v in data]
        elif isinstance(data, Serializable):
            return data.to_json()  # Serialize Serializable instances
        elif isinstance(data, SerializableEnum):
            return data.to_json()  # Use the built-in to_json of Enum
        return data  # Return unchanged if it's not a dict, list, or Serializable/SerializableEnum

    @classmethod
    def deserialize(cls, data):
        """
        Recursively deserialize data, converting JSON-encoded instances back
        into their appropriate class instances.
        """
        if isinstance(data, dict):
            print(f"Serializable.deserialize: data dict {data}")
            # Check if the dictionary has a 'class_name' key indicating a Serializable subclass
            class_name = data.get("class_name")
            if class_name and class_name in class_registry:
                # Find the class from the registry and create an instance
                specific_cls = class_registry[class_name]
                # Remove 'class_name' before passing to the constructor
                data.pop("class_name")
                instance = specific_cls.__new__(
                    specific_cls
                )  # Create an instance without calling __init__
                instance.__dict__.update(
                    cls.deserialize(data)
                )  # Update instance dictionary with deserialized data
                return instance
            else:
                # If it's a normal dictionary, recursively deserialize its items
                return {k: cls.deserialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            print(f"Serializable.deserialize: data list {data}")
            return [cls.deserialize(v) for v in data]
        elif isinstance(data, str):
            print(f"Serializable.deserialize: data str {data}")
            # If it's a string, attempt to deserialize as a SerializableEnum member
            return SerializableEnum.deserialize_if_member(data)
        return data  # Return unchanged if it's not a dict, list, or string


def is_json_dict(data):
    """
    Check if a given string represents a JSON dictionary.
    Returns the parsed dictionary if valid, or None if invalid.
    """
    if isinstance(data, str):
        try:
            parsed_data = json.loads(data)
            # Check if parsed data is a dictionary
            if isinstance(parsed_data, dict):
                return parsed_data
        except json.JSONDecodeError:
            pass
    return None  # Not a valid JSON dictionary


# Define SerializerCodec with better dynamic import handling
class SerializationJSON:

    @staticmethod
    def deserialize_dict_data(data):
        # Check if the dictionary represents a Serializable subclass by class name
        class_name = data.get("class_name")
        print(
            f"SerializationJSON.deserialize_dict_data: data dict {data}, class_name: {class_name}, in registry: {class_name in class_registry}"
        )
        if class_name in class_registry:
            cls = class_registry[class_name]
            # print(f'\t SerializationJSON.deserialize: got cls: {cls}')
            ret = cls.deserialize(data)  # Instantiate using the class method
            print(f"\t SerializationJSON.deserialize_dict_data: ret: {ret}")
            return ret
        else:
            # Printer.yellow(f'SerializationJSON.deserialize: data {data} NOT in class registry')
            return {k: SerializationJSON.deserialize(v) for k, v in data.items()}

    @staticmethod
    def serialize(data):
        """
        Recursively serialize instances of Serializable or SerializableEnum to JSON or string.
        """
        if isinstance(data, dict):
            return {k: SerializationJSON.serialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [SerializationJSON.serialize(item) for item in data]
        elif isinstance(data, Serializable):
            # If the object has a to_json method, use it
            return data.to_json()  # Serialize Serializable instances
        elif isinstance(data, SerializableEnum):
            # If the object is an Enum (SerializableEnum), use its to_json method
            return data.to_json()  # Use the built-in to_json of Enum
        elif isinstance(data, np.ndarray):
            return NumpyJson.numpy_to_json(data)
        return data  # Return unchanged if it's not a dict, list, or Serializable/SerializableEnum

    @staticmethod
    def deserialize(data):
        """
        Recursively deserialize data, handling both standard Python objects and Serializable/SerializableEnum.
        """
        if isinstance(data, dict):
            print(f"SerializationJSON.deserialize: data dict {data}")
            if NumpyJson.is_encoded(data):
                return NumpyJson.json_to_numpy(data)
            elif NumpyB64Json.is_encoded(data):
                return NumpyB64Json.json_to_numpy(data)
            else:
                return SerializationJSON.deserialize_dict_data(data)
        elif isinstance(data, list):
            print(f"SerializationJSON.deserialize: data list {data}")
            return [SerializationJSON.deserialize(item) for item in data]
        elif isinstance(data, str):
            print(f"SerializationJSON.deserialize: data str {data}")
            dict_data = is_json_dict(data)
            if dict_data is not None:
                return SerializationJSON.deserialize_dict_data(dict_data)
            else:
                print(f"SerializationJSON.deserialize: try_from_json {data}")
                # Try deserializing to SerializableEnum or other Serializable classes using from_json
                try:
                    return SerializationJSON.try_from_json(data)
                except (ValueError, KeyError):
                    return data  # If deserialization fails, return the string as is
        return data  # Return unchanged if it's not a string, dict, or list

    @staticmethod
    def try_from_json(data):
        if isinstance(data, str) and "." in data:
            class_name, enum_name = data.split(".", 1)
            try:
                # Look up the class in the registry
                enum_class = class_registry.get(class_name)
                if enum_class:
                    return enum_class[enum_name]  # Access the enum member
                else:
                    Printer.yellow(
                        f"SerializationJSON: Could not find class {class_name}, enum_name: {enum_name} in class registry"
                    )  # {class_registry}')
            except KeyError:
                Printer.red(
                    f"SerializationJSON: Could not find enum member {enum_name} in class {class_name}"
                )
        return data


# Custom encoder/decoder for saving and loading numpy arrays in json format.
# Better to be used for small numpy arrays. For large arrays, use NumpyB64Json
class NumpyJson:

    @staticmethod
    def is_encoded(data):
        return (
            isinstance(data, dict)
            and data.get("type") == "np"
            and "dtype" in data
            and "shape" in data
            and "data" in data
        )

    @staticmethod
    def numpy_to_json(data):
        if isinstance(data, np.ndarray):
            # Store data as list and save dtype and shape
            return {
                "data": data.tolist(),
                "dtype": str(data.dtype),
                "shape": data.shape,
                "type": "np",
            }
        else:
            raise TypeError(f"numpy_to_json: Expected np.ndarray, got {type(data)}")

    @staticmethod
    def json_to_numpy(data):
        # Handle None input
        if data is None:
            return None

        # Handle string input (from C++ serialization) - parse as JSON first
        if isinstance(data, str):
            try:
                data = json.loads(data)
                # After parsing, data might be None (if JSON string was "null")
                if data is None:
                    return None
            except (json.JSONDecodeError, TypeError):
                raise TypeError(f"json_to_numpy: Failed to parse string as JSON: {data}")

        if NumpyJson.is_encoded(data):
            try:
                dtype = np.dtype(data["dtype"])
                shape = tuple(data["shape"])
                array_data = data["data"]
                if isinstance(array_data, list):
                    return np.array(array_data, dtype=dtype).reshape(shape)
                else:
                    raise TypeError(
                        f"json_to_numpy: 'data' should be a list, got {type(array_data)}"
                    )
            except (TypeError, ValueError) as e:
                raise TypeError(f"json_to_numpy: Error converting data to numpy array: {e}")
        else:
            raise TypeError(
                f"json_to_numpy: Invalid input data format, expected dict with keys 'dtype', 'shape', 'data', 'type'='np'"
            )


# Custom encoder/decoder for saving and loading numpy arrays in json format by using base64 encoding.
# Better to be used for large numpy arrays.
class NumpyB64Json:

    @staticmethod
    def is_encoded(data):
        return (
            isinstance(data, dict)
            and data.get("type") == "npB64"
            and "dtype" in data
            and "shape" in data
            and "data" in data
        )

    @staticmethod
    def numpy_to_json(arr, *, order=None):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"numpy_to_json: Expected np.ndarray, got {type(arr)}")

        if arr.dtype.hasobject:
            raise TypeError("numpy_to_json: object dtype arrays are not supported")

        # Pick a stable order (preserve if possible)
        if order is None:
            order = "F" if arr.flags["F_CONTIGUOUS"] and not arr.flags["C_CONTIGUOUS"] else "C"

        raw = arr.tobytes(order=order)
        return {
            "data": base64.b64encode(raw).decode("utf-8"),
            "dtype": arr.dtype.str,  # dtype string includes endianness, more precise than str(dtype)
            "shape": list(arr.shape),  # explicit JSON-friendly type
            "order": order,
            "type": "npB64",
        }

    @staticmethod
    def json_to_numpy(data, *, writable=False):
        if data is None:
            return None

        if isinstance(data, str):
            try:
                data = json.loads(data)
                if data is None:
                    return None
            except (json.JSONDecodeError, TypeError) as e:
                raise TypeError(f"json_to_numpy: Failed to parse string as JSON: {e}")

        if not NumpyB64Json.is_encoded(data):
            raise TypeError(
                "json_to_numpy: Invalid input data format, expected dict with keys "
                "'dtype', 'shape', 'data', 'type'='npB64'"
            )

        try:
            buf = base64.b64decode(data["data"])
            dtype = np.dtype(data["dtype"])
            shape = tuple(int(x) for x in data["shape"])
            order = data.get("order", "C")

            arr = np.frombuffer(buf, dtype=dtype)
            arr = arr.reshape(shape, order=order)

            return arr.copy() if writable else arr
        except (TypeError, ValueError, binascii.Error) as e:
            raise TypeError(f"json_to_numpy: Error converting data to numpy array: {e}")

    @staticmethod
    def map_id2img_to_json(map_obj, output=None):
        if output is None:
            output = {}
        for k, v in map_obj.items():
            output[str(k)] = NumpyB64Json.numpy_to_json(v)  # normalize keys to str for JSON
        return output

    @staticmethod
    def map_id2img_from_json(map_data, output_obj=None, *, writable=False):
        if output_obj is None:
            output_obj = {}
        for k, v in map_data.items():
            output_obj[int(k)] = NumpyB64Json.json_to_numpy(v, writable=writable)
        return output_obj

    @staticmethod
    def json_to_numpy_descriptor(data, *, expected_ndim=2, writable=False):
        """
        Deserialize a descriptor array from JSON and normalize its shape.

        Args:
            data: JSON data to deserialize
            expected_ndim: Expected number of dimensions (1 for map points, 2 for frames)
            writable: Whether to return a writable copy

        Returns:
            Normalized numpy array with the expected number of dimensions
        """
        arr = NumpyB64Json.json_to_numpy(data, writable=writable)
        if arr is None:
            return None

        # Normalize shape based on expected dimensions
        if expected_ndim == 1:
            # For map point descriptors: ensure 1D (D,)
            arr = arr.flatten()
        elif expected_ndim == 2:
            # For frame descriptors: ensure 2D (N, D)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.squeeze()
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError(
                    f"Descriptor must be 2D after normalization, got shape {arr.shape}"
                )
        else:
            raise ValueError(f"Unsupported expected_ndim: {expected_ndim}")

        return arr


# ------------------------------
# Vector helpers (3D)
# ------------------------------


def cv_mat_to_json_raw(mat):
    """Convert numpy array (cv::Mat equivalent) to JSON raw format matching C++ cv_mat_to_json_raw.
    Returns dict with format: {"type": "npRaw", "dtype": "...", "shape": [...], "data": [...]}
    """
    if mat is None or (isinstance(mat, np.ndarray) and mat.size == 0):
        return None

    arr = np.ascontiguousarray(mat)

    # Map numpy dtype to C++ dtype string
    dtype_map = {
        np.uint8: "uint8",
        np.int8: "int8",
        np.uint16: "uint16",
        np.int16: "int16",
        np.int32: "int32",
        np.float32: "float32",
        np.float64: "float64",
    }
    base_dtype = dtype_map.get(arr.dtype.type, "uint8")

    # Determine shape: [rows, cols] or [rows, cols, channels]
    # C++ format: [rows, cols] for 2D, [rows, cols, channels] for 3D
    if arr.ndim == 2:
        shape = [int(arr.shape[0]), int(arr.shape[1])]
    elif arr.ndim == 1:
        shape = [int(arr.shape[0]), 1]
    elif arr.ndim == 3:
        shape = [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])]
    else:
        shape = [int(s) for s in arr.shape]

    # Flatten data
    data = arr.flatten().tolist()

    return {"type": "npRaw", "dtype": base_dtype, "shape": shape, "data": data}


def json_to_cv_mat_raw(json_data):
    """Convert JSON raw format (from cv_mat_to_json_raw) back to numpy array.
    Handles format: {"type": "npRaw", "dtype": "...", "shape": [...], "data": [...]}

    Args:
        json_data: Dict with keys "type", "dtype", "shape", "data" or None

    Returns:
        numpy.ndarray or None
    """
    if json_data is None:
        return None

    if not isinstance(json_data, dict) or json_data.get("type") != "npRaw":
        return None

    try:
        dtype_str = json_data["dtype"]
        shape = tuple(json_data["shape"])
        data = json_data["data"]
        dtype_map = {
            "uint8": np.uint8,
            "int8": np.int8,
            "uint16": np.uint16,
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
        }
        dtype = dtype_map.get(dtype_str, np.uint8)
        return np.array(data, dtype=dtype).reshape(shape)
    except (KeyError, ValueError, TypeError):
        return None


def extract_tcw_matrix_from_pose_data(pose_data):
    """Extract Tcw matrix from various pose data formats.

    Handles:
    - None: returns None
    - List/Array: direct Tcw matrix (Python/C++ aligned format)
    - Dict with "Tcw" key: extract Tcw from dict (legacy C++ CameraPose format)
    - String: parse JSON string and extract Tcw (legacy C++ CameraPose format)

    Args:
        pose_data: Can be None, list, tuple, np.ndarray, dict, or str

    Returns:
        numpy.ndarray of shape (4, 4) or None
    """
    if pose_data is None:
        return None

    # Direct array format (aligned Python and C++ serialization)
    if isinstance(pose_data, (list, tuple, np.ndarray)):
        return np.array(pose_data, dtype=np.float64)

    # Handle string format (legacy C++ CameraPose::to_json() returns JSON string)
    if isinstance(pose_data, str):
        try:
            pose_data = json.loads(pose_data)
        except (json.JSONDecodeError, ValueError):
            return None

    # Handle dict format (legacy C++ format with Tcw and covariance)
    if isinstance(pose_data, dict):
        if "Tcw" in pose_data:
            return np.array(pose_data["Tcw"], dtype=np.float64)
        # If it's a dict but no Tcw key, try to use it as array directly
        return None

    return None


def vector3_serialize(value):
    """Serialize a 3D vector as a JSON array [x, y, z] or None.
    Accepts numpy arrays of shape (3,), (3,1), or (1,3). Returns a Python list or None.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"vector3_serialize: expected 3 elements, got shape {value.shape} and size {arr.size}"
        )
    return arr.tolist()


def vector3_deserialize(val):
    """Deserialize a vector from direct JSON array format.
    - None: returns None
    - a plain list/array of 3 numbers: returns numpy array of shape (3,1)

    Returns a numpy array of shape (3,1) or None.
    """
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float64).reshape(3, 1)
        return arr
    except Exception:
        return None


# ------------------------------
# JSON helpers
# ------------------------------


# Convert inf/nan to None for JSON compatibility
def convert_inf_nan(obj):
    """Recursively convert inf and nan to None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_inf_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_inf_nan(item) for item in obj]
    elif isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None
        return obj
    return obj


# ------------------------------
# Flexible array deserialization
# ------------------------------


def deserialize_array_flexible(value, dtype=None):
    """
    Deserialize an array value from direct JSON array format.
    - None: returns None
    - String: parses as JSON (handles Python json.dumps format)
    - List/Array: directly converts to numpy array
    - Already a numpy array: returns as-is (optionally converts dtype)

    Args:
        value: The value to deserialize (can be None, str, list, tuple, or np.ndarray)
        dtype: Optional numpy dtype to convert to (e.g., np.float64, np.uint8)

    Returns:
        numpy.ndarray or None
    """
    if value is None:
        return None

    # Handle string input (Python json.dumps format) - parse as JSON
    if isinstance(value, str):
        try:
            value = json.loads(value)
            # After parsing, value might be None (if JSON string was "null")
            if value is None:
                return None
        except (json.JSONDecodeError, TypeError):
            raise TypeError(f"deserialize_array_flexible: Failed to parse string as JSON: {value}")

    if isinstance(value, (list, tuple)):
        # Direct array/list format (aligned Python and C++ serialization)
        arr = np.array(value, dtype=dtype)
        return arr
    elif isinstance(value, np.ndarray):
        # Already a numpy array
        if dtype is not None and value.dtype != dtype:
            return value.astype(dtype)
        return value
    else:
        raise TypeError(
            f"deserialize_array_flexible: Expected None, str, list, tuple, or np.ndarray, got {type(value)}"
        )
