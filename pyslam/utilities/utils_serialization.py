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

import json as json_
import ujson as json

from .utils_sys import Printer


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


class SerializableEnumEncoder(json_.JSONEncoder):
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
    def numpy_to_json(data):
        if isinstance(data, np.ndarray):
            return {
                "data": base64.b64encode(data.tobytes()).decode("utf-8"),
                "dtype": str(data.dtype),
                "shape": data.shape,
                "type": "npB64",
            }
        else:
            raise TypeError(f"numpy_to_json: Expected np.ndarray, got {type(data)}")

    @staticmethod
    def json_to_numpy(data):
        if NumpyB64Json.is_encoded(data):
            try:
                buffer = base64.b64decode(data["data"])
                shape = tuple(data["shape"])
                dtype = np.dtype(data["dtype"])
                return np.frombuffer(buffer, dtype=dtype).reshape(shape)
            except (TypeError, ValueError) as e:
                raise TypeError(f"json_to_numpy: Error converting data to numpy array: {e}")
        else:
            raise TypeError(
                f"json_to_numpy: Invalid input data format, expected dict with keys 'dtype', 'shape', 'data', 'type'='npB64'"
            )

    # Serialize map id -> dense numpy array
    @staticmethod
    def map_id2img_to_json(map_obj, output={}):
        for k, v in map_obj.items():
            output[k] = NumpyB64Json.numpy_to_json(v)
        return output

    # Deserialize map id -> dense numpy array
    @staticmethod
    def map_id2img_from_json(map_data, output_obj={}):
        for k, v in map_data.items():
            output_obj[int(k)] = NumpyB64Json.json_to_numpy(v)
        return output_obj
