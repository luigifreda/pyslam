"""
* This file is part of PYSLAM
* Adpated from adapted from https://github.com/lzx551402/contextdesc/blob/master/utils/tf.py, see the license therein.
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

import os
import sys
import importlib
import warnings  # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427

warnings.filterwarnings("ignore", category=FutureWarning)


def _register_tensorboard_stub():
    """Return TensorBoard's lightweight TensorFlow stub if full TF is unavailable."""
    try:
        import tensorboard.compat.tensorflow_stub as tf_stub
    except Exception:
        return None

    # Expose the stub through the usual TensorFlow import paths so downstream
    # libraries (e.g., torch.utils.tensorboard) see a consistent module.
    tf_stub._pyslam_tf_stub = True  # flag for downstream checks
    sys.modules["tensorflow"] = tf_stub
    if hasattr(tf_stub, "compat"):
        tf_stub.compat._pyslam_tf_stub = True
        sys.modules["tensorflow.compat"] = tf_stub.compat
        if hasattr(tf_stub.compat, "v1"):
            tf_stub.compat.v1._pyslam_tf_stub = True
            sys.modules["tensorflow.compat.v1"] = tf_stub.compat.v1
    return tf_stub


def import_tf_compat_v1():
    """
    Import TensorFlow 1.x compatibility layer.
    Works on both Linux (where tensorflow.compat.v1 can be imported directly)
    and macOS (where we need to import tensorflow first and then access compat.v1).
    
    Returns:
        tf: TensorFlow 1.x compatibility module
        
    Raises:
        RuntimeError: If TensorFlow 1.x compatibility layer cannot be accessed
    """
    # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
    # Try compat.v1 first (for newer TensorFlow versions, typically works on Linux)
    try:
        import tensorflow.compat.v1 as tf
        return tf
    except (ModuleNotFoundError, ImportError):
        pass

    def _fail(msg, exc=None):
        # Register stub so tensorboard-dependent libs do not crash, but fail for TF features.
        tf_stub = _register_tensorboard_stub()
        if tf_stub is not None:
            warnings.warn(
                msg
                + " TensorBoard stub registered so tensorboard-dependent utilities keep working, "
                  "but TensorFlow-based features will remain unavailable."
            )
        raise RuntimeError(msg) from exc

    # On macOS, the direct import path might not work, but we can import tensorflow and access compat.v1
    try:
        _tf = importlib.import_module("tensorflow")
    except Exception as exc:
        _fail(
            "TensorFlow is not installed. Please install TensorFlow 2.x with compat.v1 support "
            "(e.g., on macOS: pip install 'tensorflow-macos==2.15.*' tensorflow-metal)",
            exc,
        )

    # Get TensorFlow version for diagnostics
    tf_version = getattr(_tf, "__version__", "unknown")

    # Check if compat exists
    if not hasattr(_tf, "compat"):
        # If TensorFlow doesn't have compat, it might be an old version or special build
        # Try to use tensorflow directly (might be TF 1.x already)
        if hasattr(_tf, "ConfigProto") or hasattr(_tf, "Session"):
            return _tf
        _fail(
            f"TensorFlow {tf_version} does not have 'compat' attribute and does not appear to be TF 1.x. "
            "This typically happens with incomplete macOS installations (e.g., tensorflow-macos without the full wheel). "
            "Please ensure TensorFlow 2.x with compat.v1 support is installed (suggestion: "
            "pip install 'tensorflow-macos==2.15.*' tensorflow-metal)."
        )

    # Try to access compat.v1
    if hasattr(_tf.compat, "v1"):
        try:
            _tf.compat.v1.disable_v2_behavior()
        except Exception:
            # Even if disable_v2_behavior is missing we still try to return compat.v1
            pass
        tf = _tf.compat.v1
    else:
        tf = None

    # Verify that tf now has the expected TF 1.x APIs
    if tf is not None and (hasattr(tf, "Graph") or hasattr(tf, "Session")):
        return tf

    _fail(
        f"Failed to access TensorFlow 1.x compatibility layer (TensorFlow version: {tf_version}). "
        "Install a full TensorFlow build with compat.v1 support."
    )


# Lazy-loaded TensorFlow module
_tf_module = None


def _assert_not_stub(tf):
    """Raise a clear error if we only have the TensorBoard stub available."""
    if getattr(tf, "_pyslam_tf_stub", False):
        raise RuntimeError(
            "TensorFlow stub detected. Install a full TensorFlow package with compat.v1 "
            "support to use TensorFlow-based features (e.g., on macOS: pip install "
            "'tensorflow-macos==2.15.*' tensorflow-metal)."
        )


def _get_tf():
    """Get TensorFlow module, loading it lazily on first access."""
    global _tf_module
    if _tf_module is None:
        if False:
            import tensorflow as _tf_module
        else:
            _tf_module = import_tf_compat_v1()
    return _tf_module


def ensure_tensorflow_stub_for_tensorboard():
    """
    TensorBoard (and torch.utils.tensorboard) expects a usable tensorflow module.
    On macOS with partial installs we may only have a namespace package with no APIs,
    so we register TensorBoard's stub to avoid AttributeError crashes.
    """
    tf_mod = sys.modules.get("tensorflow")
    if tf_mod is None:
        try:
            tf_mod = importlib.import_module("tensorflow")
        except Exception:
            tf_mod = None

    if tf_mod is not None and getattr(tf_mod, "_pyslam_tf_stub", False):
        return

    if tf_mod is None or not hasattr(tf_mod, "compat") or not hasattr(tf_mod, "io"):
        _register_tensorboard_stub()


def load_frozen_model(pb_path, prefix="", print_nodes=False):
    """Load frozen model (.pb file) for testing.
    After restoring the model, operators can be accessed by
    graph.get_tensor_by_name('<prefix>/<op_name>')
    Args:
        pb_path: the path of frozen model.
        prefix: prefix added to the operator name.
        print_nodes: whether to print node names.
    Returns:
        graph: tensorflow graph definition.
    """
    tf = _get_tf()
    _assert_not_stub(tf)
    if os.path.exists(pb_path):
        with tf.io.gfile.GFile(pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=prefix)
            if print_nodes:
                for op in graph.get_operations():
                    print(op.name)
            return graph
    else:
        print("Model file does not exist", pb_path)
        exit(-1)


def recoverer(sess, model_path, meta_graph_path=None):
    """
    Recovery parameters from a pretrained model.
    Args:
        sess: The tensorflow session instance.
        model_path: Checkpoint file path.
    Returns:
        Nothing
    """
    tf = _get_tf()
    _assert_not_stub(tf)
    if meta_graph_path is None:
        restore_var = tf.global_variables()
        restorer = tf.train.Saver(restore_var)
    else:
        restorer = tf.train.import_meta_graph(meta_graph_path)
    restorer.restore(sess, model_path)


# from https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
def set_tf_logging(logging_flag):
    print("setting tf logging:", logging_flag)
    tf = _get_tf()
    if logging_flag:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        # Try to set logger level if available (TF 2.x), otherwise just use env var (TF 1.x)
        if hasattr(tf, 'get_logger'):
            tf.get_logger().setLevel("INFO")
        else:
            print("WARNING: tf.get_logger() is not available, using os.environ['TF_CPP_MIN_LOG_LEVEL'] instead")
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        # Try to set logger level if available (TF 2.x), otherwise just use env var (TF 1.x)
        if hasattr(tf, 'get_logger'):
            tf.get_logger().setLevel("ERROR")
        else:
            print("WARNING: tf.get_logger() is not available, using os.environ['TF_CPP_MIN_LOG_LEVEL'] instead")
