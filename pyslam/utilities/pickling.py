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

"""
Pickling utilities for handling multiprocessing serialization issues.

This module provides utilities to filter out unpicklable objects (like classmethod
and staticmethod) from data structures before passing them to multiprocessing.
"""

import pickle
import copy
from typing import Any, Set, Optional, List, Tuple


def find_unpicklable_objects(
    obj: Any, path: str = "root", visited: Optional[Set[int]] = None
) -> List[Tuple[str, Any, type]]:
    """
    Recursively find all classmethod and staticmethod objects in a data structure.

    Args:
        obj: Object to search
        path: Current path in the object hierarchy (for reporting)
        visited: Set of object IDs already visited (to avoid infinite recursion)

    Returns:
        List of tuples (path, object, type) for each unpicklable object found
    """
    if visited is None:
        visited = set()

    results = []
    obj_id = id(obj)
    if obj_id in visited:
        return results
    visited.add(obj_id)

    try:
        # Check if this object itself is unpicklable
        if isinstance(obj, (classmethod, staticmethod)):
            results.append((path, obj, type(obj)))
            return results

        # Check dictionaries
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path != "root" else str(key)
                results.extend(find_unpicklable_objects(value, new_path, visited))

        # Check lists and tuples
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                results.extend(find_unpicklable_objects(item, new_path, visited))

        # Check objects with __dict__
        elif hasattr(obj, "__dict__"):
            obj_dict = getattr(obj, "__dict__", {})
            for key, value in obj_dict.items():
                if isinstance(value, (classmethod, staticmethod)):
                    new_path = f"{path}.{key}" if path != "root" else key
                    results.append((new_path, value, type(value)))
                else:
                    new_path = f"{path}.{key}" if path != "root" else key
                    results.extend(find_unpicklable_objects(value, new_path, visited))

    except Exception:
        pass
    finally:
        visited.discard(obj_id)

    return results


def test_picklable(obj: Any, name: str = "object", verbose: bool = False) -> bool:
    """
    Test if an object can be pickled.

    Args:
        obj: Object to test
        name: Name of the object for error messages
        verbose: If True, print warnings for unpicklable objects

    Returns:
        True if the object can be pickled, False otherwise
    """
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        if verbose:
            from pyslam.utilities.logging import Printer

            Printer.orange(f"Object '{name}' is not picklable: {type(e).__name__}: {e}")
        return False


def filter_unpicklable_recursive(obj: Any, visited: Optional[Set[int]] = None) -> Any:
    """
    Recursively filter out classmethod and staticmethod objects from any data structure.

    This is needed because some objects might have classmethods nested in their attributes,
    which cannot be pickled by multiprocessing.

    Args:
        obj: Object to filter
        visited: Set of object IDs already visited (to avoid infinite recursion)

    Returns:
        Filtered object with classmethods/staticmethods removed
    """
    if visited is None:
        visited = set()

    # Avoid infinite recursion
    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    try:
        # Handle classmethod and staticmethod directly
        if isinstance(obj, (classmethod, staticmethod)):
            return None  # Remove it

        # Handle dictionaries
        if isinstance(obj, dict):
            filtered = {}
            for key, value in obj.items():
                filtered_value = filter_unpicklable_recursive(value, visited)
                if filtered_value is not None and not isinstance(
                    filtered_value, (classmethod, staticmethod)
                ):
                    filtered[key] = filtered_value
            return filtered

        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            filtered = []
            for item in obj:
                filtered_item = filter_unpicklable_recursive(item, visited)
                if filtered_item is not None and not isinstance(
                    filtered_item, (classmethod, staticmethod)
                ):
                    filtered.append(filtered_item)
            return type(obj)(filtered) if isinstance(obj, tuple) else filtered

        # Handle objects with __dict__
        elif hasattr(obj, "__dict__"):
            try:
                # If the object has __getstate__, trust it to handle pickling correctly
                if hasattr(obj, "__getstate__"):
                    return obj

                obj_dict = getattr(obj, "__dict__", {})
                # Check if any values are classmethods/staticmethods
                has_unpicklable = any(
                    isinstance(v, (classmethod, staticmethod)) for v in obj_dict.values()
                )
                if has_unpicklable:
                    # Try to create a copy without classmethods
                    try:
                        obj_copy = copy.copy(obj)
                        obj_copy_dict = getattr(obj_copy, "__dict__", {})
                        keys_to_remove = [
                            k
                            for k, v in obj_copy_dict.items()
                            if isinstance(v, (classmethod, staticmethod))
                        ]
                        for key in keys_to_remove:
                            try:
                                delattr(obj_copy, key)
                            except Exception:
                                pass
                        # Recursively filter nested objects in the copy
                        for key, value in obj_copy_dict.items():
                            if key not in keys_to_remove:
                                filtered_value = filter_unpicklable_recursive(value, visited)
                                if filtered_value is not None and filtered_value != value:
                                    try:
                                        setattr(obj_copy, key, filtered_value)
                                    except Exception:
                                        pass
                        return obj_copy
                    except Exception:
                        # If copying fails, return original and hope for the best
                        pass
            except Exception:
                pass
            # Return the object as-is - if it has __getstate__, that will handle it
            return obj

        # For other types, return as-is
        return obj
    except Exception:
        # If anything goes wrong, return the object as-is
        return obj
    finally:
        visited.discard(obj_id)


def ensure_picklable(obj: Any, name: str = "object", verbose: bool = False) -> Any:
    """
    Ensure an object is picklable by filtering if necessary.

    Tests the object for picklability, and if it fails, attempts to filter
    out unpicklable elements recursively.

    Args:
        obj: Object to ensure is picklable
        name: Name of the object for error messages
        verbose: If True, print warnings and diagnostic information

    Returns:
        Object that should be picklable (may be filtered version of original)
    """
    if obj is None:
        return None

    if test_picklable(obj, name, verbose=False):
        return obj

    # Object is not picklable - find what's causing it
    if verbose:
        from pyslam.utilities.logging import Printer

        Printer.orange(f"\n{'='*60}")
        Printer.orange(f"Diagnosing unpicklable object: {name}")
        Printer.orange(f"{'='*60}")

        # Find all unpicklable objects
        unpicklable_items = find_unpicklable_objects(obj, name)
        if unpicklable_items:
            Printer.orange(f"Found {len(unpicklable_items)} unpicklable item(s):")
            for path, item, item_type in unpicklable_items:
                Printer.orange(f"  - {path}: {item_type.__name__}")
                # Try to get more info about the object
                try:
                    if hasattr(item, "__func__"):
                        Printer.orange(
                            f"    Function: {item.__func__.__name__ if hasattr(item.__func__, '__name__') else 'unknown'}"
                        )
                except Exception:
                    pass
        else:
            Printer.orange("No classmethod/staticmethod found, but object still not picklable.")
            Printer.orange(f"Object type: {type(obj)}")
            Printer.orange(f"Object: {obj}")

        Printer.orange(f"{'='*60}\n")

    # Try to filter
    try:
        filtered = filter_unpicklable_recursive(obj)
        if test_picklable(filtered, f"filtered_{name}", verbose=False):
            if verbose:
                from pyslam.utilities.logging import Printer

                Printer.green(f"Successfully filtered {name}")
            return filtered
        elif verbose:
            from pyslam.utilities.logging import Printer

            Printer.orange(f"Warning: {name} still not picklable after filtering")
    except Exception as e:
        if verbose:
            from pyslam.utilities.logging import Printer

            Printer.orange(f"Warning: Failed to filter {name}: {e}")

    # Return original if filtering didn't help
    return obj
