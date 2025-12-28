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

import platform

import pyslam.config as config

import time
import traceback
import torch.multiprocessing as mp
import threading
import queue
from enum import Enum

# NOTE: Heavy graphics imports (pypangolin, glutils, OpenGL) are moved to viewer_run()
# to avoid slow startup when using multiprocessing with 'spawn' method.
# These are only needed in the child process, not in the main process.
import numpy as np


from pyslam.config_parameters import Parameters
from pyslam.slam import Map, MapStateData

from pyslam.semantics.semantic_mapping_shared import SemanticMappingShared
from pyslam.utilities.geometry import poseRt, inv_poseRt, inv_T
from pyslam.utilities.geom_trajectory import (
    TrajectoryAlignerProcessBatch,
    TrajectoryAlignerProcessIncremental,
)
from pyslam.utilities.logging import Printer
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.data_management import empty_queue, get_last_item_from_queue
from pyslam.utilities.colors import GlColors
from pyslam.utilities.waiting import wait_for_ready

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.slam.slam import Slam
    from pyslam.slam.map import Map, MapStateData


kUiWidth = 190

kDefaultSparsePointSize = 2
kDefaultDensePointSize = 2

kViewportWidth = 1024
kViewportHeight = 620

kDrawReferenceCamera = True

kMinWeightForDrawingCovisibilityEdge = Parameters.kMinWeightForDrawingCovisibilityEdge
kMaxSparseMapPointsToVisualize = Parameters.kMaxSparseMapPointsToVisualize


kAlignGroundTruthNumMinKeyframes = 10  # minimum number of keyframes to start aligning
kAlignGroundTruthMaxEveryNKeyframes = 10  # maximum number of keyframes between alignments
kAlignGroundTruthMinNumFramesPassed = 10  # minimum number of frames passed since last alignment
kAlignGroundTruthMaxEveryNFrames = 20  # maximum number of frames between alignments
kAlignGroundTruthMaxEveryTimeInterval = 3  # [s] maximum time interval between alignments

kRefreshDurationTime = 0.03  # [s]


# ========================================================
# Viz base classes
# ========================================================


class VizPointCloud:
    def __init__(self, points=None, colors=None, normalize_colors=False, reverse_colors=False):
        self.points = np.asarray(points) if points is not None else None
        self.colors = np.asarray(colors) if colors is not None else None
        if self.colors is not None and self.colors.shape[1] == 4:
            self.colors = self.colors[:, 0:3]
        if reverse_colors and self.colors is not None:
            self.colors = self.colors[:, ::-1]
        if normalize_colors and self.colors is not None:
            self.colors = self.colors / 255.0


class VizMesh:
    def __init__(
        self,
        vertices=None,
        triangles=None,
        vertex_colors=None,
        vertex_normals=None,
        normalize_colors=False,
    ):
        self.vertices = np.asarray(vertices) if vertices is not None else None
        self.triangles = np.asarray(triangles) if triangles is not None else None
        self.vertex_colors = np.asarray(vertex_colors) if vertex_colors is not None else None
        self.vertex_normals = np.asarray(vertex_normals) if vertex_normals is not None else None
        if self.vertex_colors is not None and self.vertex_colors.shape[1] == 4:
            self.vertex_colors = self.vertex_colors[:, 0:3]
        if normalize_colors and self.vertex_colors is not None:
            self.vertex_colors = self.vertex_colors / 255.0


class VizCameraImage:
    id_ = 0

    def __init__(
        self,
        image=None,
        Twc=None,
        id=None,
        scale=1.0,
        h_ratio=0.75,
        z_ratio=0.6,
        color=(0.0, 1.0, 0.0),
    ):
        self.image = image if image is not None else None
        self.Twc = Twc if Twc is not None else None
        self.scale = scale
        self.h_ratio = h_ratio
        self.z_ratio = z_ratio
        self.color = color

        if id is not None:
            self.id = id
        else:
            self.id = VizCameraImage.id_
            VizCameraImage.id_ += 1


class ViewerCurrentFrameData:
    def __init__(self):
        self.cur_frame_id = None
        self.cur_pose = None
        self.cur_pose_timestamp = None
        self.predicted_pose = None
        self.reference_pose = None


class Viewer3DMapInput:
    def __init__(self):
        self.cur_frame_data = ViewerCurrentFrameData()
        self.map_data: "MapStateData" | None = (
            None  # map state data in the form of a set of data arrays for the viewer
        )
        self.gt_trajectory = None
        self.gt_timestamps = None
        self.align_gt_with_scale = False

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class Viewer3DDenseInput:
    def __init__(self):
        self.point_cloud = None
        self.mesh = None
        self.camera_images = []  # list of VizCameraImage objects


class Viewer3DVoInput:
    def __init__(self):
        self.poses = []
        self.pose_timestamps = []
        self.traj3d_est = []  # estimated trajectory
        self.traj3d_gt = []  # ground truth trajectory


class Viewer3DCameraTrajectoriesInput:
    def __init__(self, build_trajectory_line=False):
        self.build_trajectory_line = build_trajectory_line
        self.camera_trajectories = (
            []
        )  # a list of camera trajectories (a camera trajectory is an array of np 4x4 matrices)
        self.camera_trajectory_lines = []
        self.camera_colors = []

    def reset(self):
        self.camera_trajectories = []
        self.camera_trajectory_lines = []
        self.camera_colors = []

    def add(self, camera_pose_array, camera_color=None):
        self.camera_trajectories.append(camera_pose_array)
        if self.build_trajectory_line:
            trajectory_line = np.asarray([c[:3, 3] for c in camera_pose_array]).reshape(-1, 3)
            self.camera_trajectory_lines.append(trajectory_line)
        if camera_color is None:
            unused_colors = [c for c in GlColors.get_colors() if c not in self.camera_colors]
            camera_color = (
                unused_colors[0] if len(unused_colors) > 0 else GlColors.get_random_color()
            )
        self.camera_colors.append(camera_color)


# ========================================================
# SlamDrawerThread class
# ========================================================


class SlamDrawingTaskType(Enum):
    """Enum for different types of SLAM drawing tasks"""

    DRAW_SLAM_MAP = "draw_slam_map"
    DRAW_DENSE_MAP = "draw_dense_map"


class SlamDrawingTask:
    def __init__(
        self,
        task_type: SlamDrawingTaskType,
        slam: "Slam",
        viewer_cur_frame_data: ViewerCurrentFrameData | None = None,
    ):
        self.task_type = task_type
        self.slam = slam
        self.viewer_cur_frame_data = viewer_cur_frame_data


class SlamDrawerThread:
    """
    Threaded drawer that processes SLAM drawing requests asynchronously.
    This allows the main SLAM loop to continue without waiting for expensive
    data preparation operations like get_data_arrays_for_drawing().

    Uses a thread-safe reference to slam object (no pickling needed since we're using threading).
    """

    kSleepTime = 0.005  # [s] sleep time between drawing requests

    def __init__(self, viewer3D: "Viewer3D"):
        self.viewer3D = viewer3D

        # Thread-safe queue for drawing requests (only task type, no slam object)
        # Use maxsize=2 to keep only the latest requests (drop old ones)
        self.request_queue = queue.Queue(maxsize=2)

        # Thread-safe reference to slam object (updated atomically)
        self._slam_lock = threading.Lock()
        self._slam_ref = None

        self.is_running = threading.Event()
        self.is_running.set()

        # Start drawing thread
        self.draw_thread = threading.Thread(target=self.run, daemon=True, name="SlamDrawerThread")
        self.draw_thread.start()

    def __getstate__(self):
        """Exclude non-picklable objects (locks, threads) from pickling"""
        state = self.__dict__.copy()
        # Remove non-picklable objects
        state.pop("_slam_lock", None)
        state.pop("draw_thread", None)
        state.pop("is_running", None)
        state.pop("request_queue", None)
        return state

    def __setstate__(self, state):
        """Restore state and recreate non-picklable objects"""
        self.__dict__.update(state)
        # Recreate non-picklable objects
        self._slam_lock = threading.Lock()
        self.request_queue = queue.Queue(maxsize=2)
        self.is_running = threading.Event()
        self.is_running.set()
        # Restart the thread if viewer3D is available
        if hasattr(self, "viewer3D") and self.viewer3D is not None:
            self.draw_thread = (
                threading.Thread(target=self.run, daemon=True, name="SlamDrawerThread")
                if self.draw_thread is None
                else None
            )
            self.draw_thread.start()

    def run(self):
        """Worker thread that processes drawing requests"""
        while self.is_running.is_set():
            try:
                # Get request with timeout to allow checking is_running
                try:
                    task = self.request_queue.get(timeout=0.1)
                    task_type = task.task_type
                except queue.Empty:
                    continue

                # Get current slam reference (thread-safe)
                with self._slam_lock:
                    slam = self._slam_ref

                if slam is None:
                    continue  # Skip if no slam reference set

                # Process the drawing request
                if task_type == SlamDrawingTaskType.DRAW_SLAM_MAP:
                    viewer_cur_frame_data = task.viewer_cur_frame_data
                    self.viewer3D._draw_slam_map_impl(slam, viewer_cur_frame_data)
                elif task_type == SlamDrawingTaskType.DRAW_DENSE_MAP:
                    self.viewer3D._draw_dense_map_impl(slam)
                else:
                    time.sleep(self.kSleepTime)

                self.request_queue.task_done()
            except Exception as e:
                Printer.red(f"SlamDrawerThread: error in draw worker: {e}")
                traceback.print_exc()
        print(f"SlamDrawerThread: run: closed")

    def request_draw(self, task: SlamDrawingTask):
        """Request a drawing operation (non-blocking, no pickling)"""
        if not self.is_running.is_set():
            return

        # Update slam reference atomically (no pickling needed)
        with self._slam_lock:
            self._slam_ref = task.slam

        try:
            # Try to put the request, but don't block if queue is full
            # This drops old requests if the queue is full, keeping only the latest
            try:
                self.request_queue.put_nowait(task)
            except queue.Full:
                # Queue is full, try to get and discard old request, then put new one
                try:
                    self.request_queue.get_nowait()
                    self.request_queue.put_nowait(task)
                except queue.Empty:
                    pass  # Queue became empty between checks
        except Exception as e:
            Printer.red(f"SlamDrawerThread: error requesting draw: {e}")

    def quit(self):
        """Stop the drawing thread"""
        self.is_running.clear()
        if self.draw_thread.is_alive():
            self.draw_thread.join(timeout=1.0)
        print(f"SlamDrawerThread: closed")


# ========================================================
# Viewer3D class
# ========================================================


class Viewer3D(object):
    def __init__(self, scale=0.1):
        self.scale = scale

        self.map_state: Viewer3DMapInput | None = None
        self.vo_state: Viewer3DVoInput | None = None
        self.dense_state: Viewer3DDenseInput | None = None
        self.camera_trajectories_state: Viewer3DCameraTrajectoriesInput | None = None

        self.gt_trajectory = None
        self.gt_timestamps = None
        self.align_gt_with_scale = False

        # TODO(dvdmc): to customize the visualization from the UI, we need to create mp variables accesible from the refresh to the viewer thread
        # We would need a query word and a heatmap scale.

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.qmap = self.mp_manager.Queue()
        self.qvo = self.mp_manager.Queue()
        self.qdense = self.mp_manager.Queue()
        self.qcams = self.mp_manager.Queue()

        self._is_running = mp.Value("i", 0)
        self._is_looping = mp.Value("i", 0)
        self._is_paused = mp.Value("i", 0)
        self._is_closed = mp.Value("i", 0)

        self._is_map_save = mp.Value("i", 0)
        self._is_bundle_adjust = mp.Value("i", 0)
        self._do_step = mp.Value("i", 0)
        self._do_reset = mp.Value("i", 0)
        self._is_draw_features_with_radius = mp.Value("i", 0)
        self._is_draw_instance_colors = mp.Value("i", 0)

        self._is_gt_set = mp.Value("i", 0)
        self.alignment_gt_data_queue = (
            self.mp_manager.Queue()
        )  # used by slam_plot_drawer.py to draw GT alignment data
        self.aligner_input_queue = (
            self.mp_manager.Queue()
        )  # used to pass data to the trajectory aligner
        self.aligner_output_queue = self.mp_manager.Queue()  # used to get aligner output
        self.is_aligner_running = mp.Value("i", 0)
        self.trajectory_aligner = None

        # Create threaded drawer for asynchronous drawing operations
        self.slam_drawer_thread = None
        if Parameters.kViewerDrawSlamMapOnSeparateThread:
            self.slam_drawer_thread = SlamDrawerThread(self)

        self.vp = mp.Process(
            target=self.viewer_run,
            args=(
                self.qmap,
                self.qvo,
                self.qdense,
                self.qcams,
                self._is_running,
                self._is_looping,
                self._is_paused,
                self._is_closed,
                self._is_map_save,
                self._is_bundle_adjust,
                self._do_step,
                self._do_reset,
                self._is_draw_features_with_radius,
                self._is_draw_instance_colors,
                self._is_gt_set,
                self.alignment_gt_data_queue,
                self.aligner_input_queue,
                self.aligner_output_queue,
                self.is_aligner_running,
            ),
        )
        self.vp.daemon = True
        self.vp.start()

    # NOTE: When multiprocessing spawns a new process, it pickles the Viewer3D instance because viewer_run is a bound method.
    # To avoid pickling problems, we need to exclude the non-picklable objects from the state.
    def __getstate__(self):
        """Exclude non-picklable objects from pickling (for multiprocessing)"""
        state = self.__dict__.copy()
        # Exclude slam_drawer_thread (contains locks, threads, queues - not picklable)
        # This is only used in the main process, not in the child process
        if self.slam_drawer_thread is not None:
            state.pop("slam_drawer_thread", None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling"""
        self.__dict__.update(state)
        # slam_drawer_thread will be None after unpickling, but that's OK
        # since it's only needed in the main process, not the child process

    def set_gt_trajectory(self, gt_trajectory, gt_timestamps, align_with_scale=False):
        if gt_trajectory is None or gt_timestamps is None:
            Printer.yellow("Viewer3D: set_gt_trajectory: gt_trajectory or gt_timestamps is None")
            return
        if len(gt_timestamps) > 0:
            self.gt_trajectory = gt_trajectory
            self.gt_timestamps = gt_timestamps
            self.align_gt_with_scale = align_with_scale
            self._is_gt_set.value = 0
            print(f"Viewer3D: Groundtruth shape: {gt_trajectory.shape}")

            trajectory_aligner_class = TrajectoryAlignerProcessBatch
            # trajectory_aligner_class = TrajectoryAlignerProcessIncremental  # experimental
            self.trajectory_aligner = trajectory_aligner_class(
                input_queue=self.aligner_input_queue,
                output_queue=self.aligner_output_queue,
                is_running_flag=self.is_aligner_running,
                gt_trajectory=self.gt_trajectory,
                gt_timestamps=self.gt_timestamps,
                find_scale=self.align_gt_with_scale,
                compute_align_error=True,  # we need it for the slam plot drawer to draw the alignment errors
            )
            self.trajectory_aligner.start()
            Printer.blue(f"Viewer3D: set_gt_trajectory - trajectory aligner started")

    def quit(self):
        print("Viewer3D: quitting...")
        # Stop the drawer thread first
        if hasattr(self, "slam_drawer_thread") and self.slam_drawer_thread is not None:
            self.slam_drawer_thread.quit()
        if self._is_running.value == 1:
            self._is_running.value = 0
        if self._is_looping.value == 1:
            self._is_looping.value = 0
        if self.vp.is_alive():
            self.vp.join()
        # if self.is_aligner_running.value == 1:
        #     self.is_aligner_running.value = 0
        # if self.trajectory_aligner and self.trajectory_aligner.is_alive():
        #     self.trajectory_aligner.join()
        if self.trajectory_aligner:
            self.trajectory_aligner.quit()
        # pangolin.Quit()
        print("Viewer3D: done")

    def wait_for_ready(self, timeout=None):
        wait_for_ready(self.is_ready, "Viewer3D", timeout)
        Printer.green("Viewer3D: ready")

    def is_ready(self):
        return self._is_running.value == 1 and self._is_looping.value == 1

    def is_running(self):
        return self._is_running.value == 1

    def is_closed(self):
        return self._is_closed.value == 1

    def is_paused(self):
        return self._is_paused.value == 1

    def is_map_save(self):
        is_map_save = self._is_map_save.value == 1
        if is_map_save:
            self._is_map_save.value = 0
        return is_map_save

    def is_bundle_adjust(self):
        is_bundle_adjust = self._is_bundle_adjust.value == 1
        if is_bundle_adjust:
            self._is_bundle_adjust.value = 0
        return is_bundle_adjust

    def do_step(self):
        do_step = self._do_step.value == 1
        if do_step:
            self._do_step.value = 0
        return do_step

    def reset(self):
        do_reset = self._do_reset.value == 1
        if do_reset:
            self._do_reset.value = 0
        return do_reset

    def is_draw_features_with_radius(self):
        return self._is_draw_features_with_radius.value == 1

    def is_draw_instance_colors(self):
        return self._is_draw_instance_colors.value == 1

    def viewer_run(
        self,
        qmap,
        qvo,
        qdense,
        qcams,
        is_running,
        is_looping,
        is_paused,
        is_closed,
        is_map_save,
        is_bundle_adjust,
        do_step,
        do_reset,
        is_draw_features_with_radius,
        is_draw_instance_colors,
        is_gt_set,
        alignment_gt_data_queue,
        aligner_input_queue,
        aligner_output_queue,
        is_aligner_running,
    ):
        # Import heavy graphics libraries here (in the child process) to avoid slow startup
        # when using multiprocessing with 'spawn' method. This way they're only loaded
        # when the viewer process actually starts, not when the module is imported.
        # These imports are made available to other methods (viewer_init, viewer_refresh)
        # by setting them in the module's global namespace.
        import pypangolin as pangolin
        import glutils
        import OpenGL.GL as gl

        # Make imports available to other methods in this class by setting in module globals
        # This allows viewer_init() and viewer_refresh() to access these modules when called
        globals()["pangolin"] = pangolin
        globals()["glutils"] = glutils
        globals()["gl"] = gl

        self.viewer_init(kViewportWidth, kViewportHeight)
        is_running.value = 1
        # init local vars for the the process
        self.thread_gt_trajectory = None
        self.thread_gt_trajectory_aligned = None
        self.thread_gt_trajectory_aligned_associated = None
        self.thread_gt_timestamps = None
        self.thread_gt_associated_traj = None
        self.thread_est_associated_traj = None
        self.thread_align_gt_with_scale = False
        self.thread_gt_aligned = False
        self.thread_last_num_poses_gt_was_aligned = 0
        self.thread_last_frame_id_gt_was_aligned = 0
        self.thread_last_time_gt_was_aligned = time.time()
        self.thread_alignment_gt_data_queue = alignment_gt_data_queue  # used by slam_plot_drawer.py

        is_looping.value = 1
        while not pangolin.ShouldQuit() and (is_running.value == 1):
            ts = time.time()
            try:
                self.viewer_refresh(
                    qmap,
                    qvo,
                    qdense,
                    qcams,
                    is_paused,
                    is_map_save,
                    is_bundle_adjust,
                    do_step,
                    do_reset,
                    is_draw_features_with_radius,
                    is_draw_instance_colors,
                    is_gt_set,
                    aligner_input_queue,
                    aligner_output_queue,
                    is_aligner_running,
                )
                sleep = (time.time() - ts) - kRefreshDurationTime
                if sleep > 0:
                    time.sleep(sleep)
            except Exception as e:
                Printer.red(f"Viewer3D: viewer_run - error: {e}")
                traceback.print_exc()

        empty_queue(qmap)  # empty the queue before exiting
        empty_queue(qvo)  # empty the queue before exiting
        empty_queue(qdense)  # empty the queue before exiting
        empty_queue(qcams)
        is_running.value = 0
        is_closed.value = 1
        print("Viewer3D: loop exit...")

    def viewer_init(self, w, h):
        # pangolin.ParseVarsFile('app.cfg')

        pangolin.CreateWindowAndBind("Map Viewer", w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        viewpoint_x = 0 * self.scale
        viewpoint_y = -40 * self.scale
        viewpoint_z = -80 * self.scale
        viewpoint_f = 1000

        self.proj = pangolin.ProjectionMatrix(
            w, h, viewpoint_f, viewpoint_f, w // 2, h // 2, 0.1, 5000
        )
        self.look_view = pangolin.ModelViewLookAt(
            viewpoint_x, viewpoint_y, viewpoint_z, 0, 0, 0, 0, -1, 0
        )
        self.scam = pangolin.OpenGlRenderState(self.proj, self.look_view)
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, kUiWidth / w, 1.0, -w / h)
        self.dcam.SetHandler(pangolin.Handler3D(self.scam))

        self.panel = pangolin.CreatePanel("ui")
        self.panel.SetBounds(0.0, 1.0, 0.0, kUiWidth / w)

        self.do_follow = True
        self.is_grid = True
        self.is_following = True

        self.is_draw_cameras = True
        self.is_draw_covisibility = True
        self.is_draw_spanning_tree = True
        self.is_draw_loops = True
        self.is_draw_dense = True
        self.is_draw_sparse = True
        self.is_draw_semantic_colors = False

        self.draw_wireframe = False

        self.draw_gt_changed = False
        self.draw_gt = False
        self.draw_gt_associations = False

        # self.button = pangolin.VarBool('ui.Button', value=False, toggle=False)

        self.checkboxFollow = pangolin.VarBool("ui.Follow", value=True, toggle=True)
        self.checkboxCams = pangolin.VarBool("ui.Draw Cameras", value=True, toggle=True)
        self.checkboxCovisibility = pangolin.VarBool(
            "ui.Draw Covisibility", value=True, toggle=True
        )
        self.checkboxSpanningTree = pangolin.VarBool("ui.Draw Tree", value=True, toggle=True)
        self.checkboxLoops = pangolin.VarBool("ui.Draw Loops", value=True, toggle=True)
        self.checkboxGT = pangolin.VarBool("ui.Draw Ground Truth", value=False, toggle=True)
        self.checkboxGTassociations = pangolin.VarBool(
            "ui.Draw GT Associations", value=False, toggle=True
        )
        self.checkboxPredicted = pangolin.VarBool("ui.Draw Predicted", value=False, toggle=True)
        self.checkboxFovCenters = pangolin.VarBool("ui.Draw Fov Centers", value=False, toggle=True)
        self.checkboxDrawSparseCloud = pangolin.VarBool(
            "ui.Draw Sparse Map", value=True, toggle=True
        )
        self.checkboxDrawDenseCloud = pangolin.VarBool("ui.Draw Dense Map", value=True, toggle=True)
        self.checkboxColorSemantics = pangolin.VarBool(
            "ui.Color Semantics", value=False, toggle=True
        )
        self.checkboxInstanceColors = pangolin.VarBool(
            "ui.Instance Colors", value=False, toggle=True
        )
        self.checkboxGrid = pangolin.VarBool("ui.Grid", value=True, toggle=True)
        self.checkboxDrawFeaturesWithRadius = pangolin.VarBool(
            "ui.Features Radius", value=False, toggle=True
        )
        self.checkboxPause = pangolin.VarBool("ui.Pause", value=False, toggle=True)

        self.buttonSave = pangolin.VarBool("ui.Save", value=False, toggle=False)
        self.buttonStep = pangolin.VarBool("ui.Step", value=False, toggle=False)
        self.buttonReset = pangolin.VarBool("ui.Reset", value=False, toggle=False)
        self.buttonBA = pangolin.VarBool("ui.Bundle Adjust", value=False, toggle=False)

        # self.float_slider = pangolin.VarFloat('ui.Float', value=3, min=0, max=5)
        # self.float_log_slider = pangolin.VarFloat('ui.Log_scale var', value=3, min=1, max=1e4, logscale=True)
        self.sparsePointSizeSlider = pangolin.VarInt(
            "ui.Sparse Point Size", value=kDefaultSparsePointSize, min=1, max=10
        )
        self.densePointSizeSlider = pangolin.VarInt(
            "ui.Dense Point Size", value=kDefaultDensePointSize, min=1, max=10
        )
        self.checkboxWireframe = pangolin.VarBool("ui.Mesh Wireframe", value=False, toggle=True)

        self.sparsePointSize = self.sparsePointSizeSlider.Get()
        self.densePointSize = self.densePointSizeSlider.Get()

        self.Twc = pangolin.OpenGlMatrix()
        self.Twc.SetIdentity()
        # print("self.Twc.m",self.Twc.m)

        self.camera_images = (
            glutils.CameraImages()
        )  # current buffer of camera images update by draw_dense_map()

    def viewer_refresh(
        self,
        qmap,
        qvo,
        qdense,
        qcams,
        is_paused,
        is_map_save,
        is_bundle_adjust,
        do_step,
        do_reset,
        is_draw_features_with_radius,
        is_draw_instance_colors,
        is_gt_set,
        aligner_input_queue,
        aligner_output_queue,
        is_aligner_running,
    ):

        # NOTE: take the last elements in the queues

        last_map_state = get_last_item_from_queue(qmap)
        if last_map_state is not None:
            self.map_state = last_map_state

        last_vo_state = get_last_item_from_queue(qvo)
        if last_vo_state is not None:
            self.vo_state = last_vo_state

        last_dense_state = get_last_item_from_queue(qdense)
        if last_dense_state is not None:
            self.dense_state = last_dense_state

            if self.dense_state is not None:
                # Update the dense state with the last received dense state
                # update the camera images buffer
                self.camera_images.clear()
                for cam in self.dense_state.camera_images:
                    if not isinstance(cam, VizCameraImage):
                        Printer.red(
                            f"Viewer3D: viewer_refresh - camera_images should be a list of VizCameraImage objects - found cam of type: {type(cam)}"
                        )
                        continue
                    # print(f'Viewer3D: viewer_refresh - adding camera image with id: {cam.id}, scale: {cam.scale}, Twc: {cam.Twc}, image.shape: {cam.image.shape}, image.dtype: {cam.image.dtype}')
                    self.camera_images.add(
                        image=cam.image,
                        pose=cam.Twc,
                        id=cam.id,
                        scale=cam.scale,
                        h_ratio=cam.h_ratio,
                        z_ratio=cam.z_ratio,
                        color=cam.color,
                    )

        last_camera_trajectories_state = get_last_item_from_queue(qcams)
        if last_camera_trajectories_state is not None:
            self.camera_trajectories_state = last_camera_trajectories_state

        self.do_follow = self.checkboxFollow.Get()
        self.is_grid = self.checkboxGrid.Get()

        is_draw_features_with_radius.value = self.checkboxDrawFeaturesWithRadius.Get()
        is_draw_instance_colors.value = self.checkboxInstanceColors.Get()

        self.is_draw_cameras = self.checkboxCams.Get()
        self.is_draw_covisibility = self.checkboxCovisibility.Get()
        self.is_draw_spanning_tree = self.checkboxSpanningTree.Get()
        self.is_draw_loops = self.checkboxLoops.Get()
        self.draw_gt_changed = (
            self.draw_gt != self.checkboxGT.Get()
            or self.draw_gt_associations != self.checkboxGTassociations.Get()
        )
        self.draw_gt = self.checkboxGT.Get()
        self.draw_gt_associations = self.checkboxGTassociations.Get()
        self.draw_predicted = self.checkboxPredicted.Get()
        self.draw_fov_centers = self.checkboxFovCenters.Get()
        self.draw_wireframe = self.checkboxWireframe.Get()
        self.is_draw_dense = self.checkboxDrawDenseCloud.Get()
        self.is_draw_sparse = self.checkboxDrawSparseCloud.Get()
        self.is_draw_semantic_colors = self.checkboxColorSemantics.Get()

        # if pangolin.Pushed(self.checkboxPause):
        if self.checkboxPause.Get():
            is_paused.value = 1
        else:
            # if is_paused.value == 1:                 # down-front
            #     if self.camera_trajectories_state is not None:
            #         self.camera_trajectories_state.reset()
            is_paused.value = 0

        if pangolin.Pushed(self.buttonSave):
            self.checkboxPause.SetVal(True)
            is_paused.value = 1
            is_map_save.value = 1

        if pangolin.Pushed(self.buttonBA):
            self.checkboxPause.SetVal(True)
            is_paused.value = 1
            is_bundle_adjust.value = 1

        if pangolin.Pushed(self.buttonStep):
            if not is_paused.value:
                self.checkboxPause.SetVal(True)
                is_paused.value = 1
            do_step.value = 1

        if pangolin.Pushed(self.buttonReset):
            # if not is_paused.value:
            #     self.checkboxPause.SetVal(True)
            #     is_paused.value = 1
            do_reset.value = 1

        # self.sparsePointSizeSlider.SetVal(int(self.float_slider))
        self.sparsePointSize = self.sparsePointSizeSlider.Get()
        self.densePointSize = self.densePointSizeSlider.Get()

        if self.do_follow and self.is_following:
            self.scam.Follow(self.Twc, True)
        elif self.do_follow and not self.is_following:
            self.scam.SetModelViewMatrix(self.look_view)
            self.scam.Follow(self.Twc, True)
            self.is_following = True
        elif not self.do_follow and self.is_following:
            self.is_following = False

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)

        self.dcam.Activate(self.scam)

        if self.is_grid:
            # Viewer3D.drawPlane(scale=self.scale)
            glutils.DrawPlane(num_divs=200, div_size=10, scale=self.scale)

        # ==============================
        # draw map
        if self.map_state is not None:
            map_data = self.map_state.map_data
            cur_frame_data = self.map_state.cur_frame_data

            if not is_gt_set.value and self.map_state.gt_trajectory is not None:
                self.thread_gt_trajectory = np.asarray(self.map_state.gt_trajectory)
                self.thread_gt_trajectory_aligned = self.thread_gt_trajectory.copy()
                self.thread_gt_timestamps = np.asarray(self.map_state.gt_timestamps)
                self.thread_align_gt_with_scale = self.map_state.align_gt_with_scale
                is_gt_set.value = 1

            if cur_frame_data.cur_pose is not None:
                # draw current pose in blue
                cur_pose = cur_frame_data.cur_pose.copy()
                gl.glColor3f(0.0, 0.0, 1.0)
                gl.glLineWidth(2)
                glutils.DrawCamera(cur_pose, self.scale)
                gl.glLineWidth(1)
                self.updateTwc(cur_pose)

            if self.draw_predicted and cur_frame_data.predicted_pose is not None:
                # draw predicted pose in red
                gl.glColor3f(1.0, 0.0, 0.0)
                glutils.DrawCamera(cur_frame_data.predicted_pose, self.scale)

            if self.draw_fov_centers and len(map_data.fov_centers) > 0:
                # draw keypoints with their color
                gl.glPointSize(5)
                # gl.glColor3f(1.0, 0.0, 0.0)
                glutils.DrawPoints(map_data.fov_centers, map_data.fov_centers_colors)

            if self.thread_gt_timestamps is not None:
                if self.draw_gt:
                    # align the gt to the estimated trajectory every 'kAlignGroundTruthMaxEveryNKeyframes' frames;
                    # the more estimated frames we have the better the alignment!
                    num_kfs = len(map_data.poses)
                    condition1 = (
                        time.time() - self.thread_last_time_gt_was_aligned
                    ) > kAlignGroundTruthMaxEveryTimeInterval
                    condition2 = (
                        len(map_data.poses)
                        > kAlignGroundTruthMaxEveryNKeyframes
                        + self.thread_last_num_poses_gt_was_aligned
                    )
                    condition3 = (
                        cur_frame_data.cur_frame_id
                        > kAlignGroundTruthMaxEveryNFrames
                        + self.thread_last_frame_id_gt_was_aligned
                    )
                    condition4 = (
                        cur_frame_data.cur_frame_id - self.thread_last_frame_id_gt_was_aligned
                        > kAlignGroundTruthMinNumFramesPassed
                    )

                    # Here we design if we should align the gt trajectory with the estimated trajectory
                    if (self._is_running and self.is_aligner_running.value == 1) and (
                        (
                            num_kfs > kAlignGroundTruthNumMinKeyframes
                            and condition1
                            or condition2
                            or condition3
                            and condition4
                        )
                        or self.draw_gt_changed
                    ):
                        try:

                            Printer.blue(
                                f"Viewer3D: viewer_refresh - aligning gt trajectory, c1: {condition1}, c2: {condition2}, c3: {condition3}, c4: {condition4}"
                            )

                            self.thread_last_time_gt_was_aligned = time.time()
                            self.thread_last_num_poses_gt_was_aligned = len(map_data.poses)
                            self.thread_last_frame_id_gt_was_aligned = cur_frame_data.cur_frame_id

                            estimated_trajectory = np.asarray(
                                [pose[0:3, 3] for i, pose in enumerate(map_data.poses)],
                                dtype=float,
                            )

                            self.aligner_input_queue.put(
                                (map_data.pose_timestamps, estimated_trajectory)
                            )

                            aligner_output = get_last_item_from_queue(self.aligner_output_queue)
                            if aligner_output is not None:
                                T_gt_est, error, alignment_gt_data = aligner_output
                                self.thread_alignment_gt_data_queue.put(alignment_gt_data)

                                print(
                                    f"Viewer3D: viewer_refresh - align gt with scale: {self.thread_align_gt_with_scale}, RMS error: {error}"
                                )
                                self.thread_gt_aligned = True

                                self.thread_gt_associated_traj = alignment_gt_data.gt_t_wi
                                self.thread_est_associated_traj = alignment_gt_data.estimated_t_wi
                                # all gt data aligned to the estimated
                                self.thread_gt_trajectory_aligned = (
                                    alignment_gt_data.gt_trajectory_aligned
                                )
                                # only the associated gt samples aligned to the estimated samples
                                self.thread_gt_trajectory_aligned_associated = (
                                    alignment_gt_data.gt_trajectory_aligned_associated
                                )
                        except Exception as e:
                            print(f"Viewer3D: viewer_refresh - align_gt_with_svd failed: {e}")

                    if self.thread_gt_aligned:
                        gl.glLineWidth(1)
                        gl.glColor3f(1.0, 0.0, 0.0)
                        glutils.DrawTrajectory(self.thread_gt_trajectory_aligned, 2)

                        if self.draw_gt_associations:
                            num_associations = len(self.thread_est_associated_traj)
                            is_draw_gt_associations_ok = (
                                self.thread_gt_trajectory_aligned_associated is not None
                            ) and (
                                len(self.thread_gt_trajectory_aligned_associated)
                                == num_associations
                            )
                            if num_associations > 0 and is_draw_gt_associations_ok:
                                # visualize the data associations between the aligned gt and the estimated trajectory
                                # gt_to_est_lines = [[*self.thread_gt_trajectory_aligned_associated[i], *self.thread_est_associated_traj[i]] for i in range(num_associations)]
                                gl.glLineWidth(2)
                                gl.glColor3f(0.5, 1.0, 1.0)
                                # glutils.DrawLines(gt_to_est_lines,2)
                                glutils.DrawLines2(
                                    self.thread_gt_trajectory_aligned_associated,
                                    self.thread_est_associated_traj,
                                    2,
                                )
                                gl.glLineWidth(1)

            if len(map_data.poses) > 1:
                # draw keyframe poses in green
                if self.is_draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    glutils.DrawCameras(map_data.poses, self.scale)

            if self.is_draw_sparse and len(map_data.points) > 0:
                # draw keypoints with their color
                gl.glPointSize(self.sparsePointSize)
                # gl.glColor3f(1.0, 0.0, 0.0)
                if self.is_draw_semantic_colors:
                    colors = map_data.semantic_colors
                else:
                    colors = map_data.colors
                glutils.DrawPoints(map_data.points, colors)

            if cur_frame_data.reference_pose is not None and kDrawReferenceCamera:
                # draw predicted pose in purple
                gl.glColor3f(0.5, 0.0, 0.5)
                gl.glLineWidth(2)
                glutils.DrawCamera(cur_frame_data.reference_pose, self.scale)
                gl.glLineWidth(1)

            if len(map_data.covisibility_graph) > 0:
                if self.is_draw_covisibility:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 1.0, 0.0)
                    glutils.DrawLines(map_data.covisibility_graph, 3)

            if len(map_data.spanning_tree) > 0:
                if self.is_draw_spanning_tree:
                    gl.glLineWidth(1)
                    gl.glColor3f(0.0, 0.0, 1.0)
                    glutils.DrawLines(map_data.spanning_tree, 3)

            if len(map_data.loops) > 0:
                if self.is_draw_loops:
                    gl.glLineWidth(2)
                    gl.glColor3f(0.5, 0.0, 0.5)
                    glutils.DrawLines(map_data.loops, 3)
                    gl.glLineWidth(1)

        # ==============================
        # draw dense stuff
        if self.dense_state is not None:
            if self.is_draw_dense:
                if self.dense_state.mesh is not None:
                    vertices = self.dense_state.mesh[0]
                    triangles = self.dense_state.mesh[1]
                    colors = self.dense_state.mesh[2]
                    glutils.DrawMesh(vertices, triangles, colors, self.draw_wireframe)
                elif self.dense_state.point_cloud is not None:
                    pc_points = self.dense_state.point_cloud[0]
                    pc_colors = self.dense_state.point_cloud[1]
                    pc_semantic_colors = self.dense_state.point_cloud[2]
                    gl.glPointSize(self.densePointSize)
                    if self.is_draw_semantic_colors and pc_semantic_colors is not None:
                        glutils.DrawPoints(pc_points, pc_semantic_colors)
                    else:
                        glutils.DrawPoints(pc_points, pc_colors)

        if self.camera_images.size() > 0:
            self.camera_images.draw()

        # ==============================
        # draw vo
        if self.vo_state is not None:
            if self.vo_state.poses.shape[0] >= 2:
                # draw poses in green
                if self.is_draw_cameras:
                    gl.glColor3f(0.0, 1.0, 0.0)
                    glutils.DrawCameras(self.vo_state.poses, self.scale)

            if self.vo_state.poses.shape[0] >= 1:
                # draw current pose in blue
                gl.glColor3f(0.0, 0.0, 1.0)
                current_pose = self.vo_state.poses[-1:]
                glutils.DrawCameras(current_pose, self.scale)
                self.updateTwc(current_pose[0])

            if self.vo_state.traj3d_est.shape[0] != 0:
                # draw blue estimated trajectory
                gl.glPointSize(self.sparsePointSize)
                gl.glColor3f(0.0, 0.0, 1.0)
                glutils.DrawLine(self.vo_state.traj3d_est)

            if self.vo_state.traj3d_gt.shape[0] != 0:
                # draw red ground-truth trajectory
                gl.glPointSize(self.sparsePointSize)
                gl.glColor3f(1.0, 0.0, 0.0)
                glutils.DrawLine(self.vo_state.traj3d_gt)

        # ==============================
        # draw camera pose arrays
        if self.camera_trajectories_state is not None:
            if len(self.camera_trajectories_state.camera_trajectories) > 0:
                for i, camera_trajectory in enumerate(
                    self.camera_trajectories_state.camera_trajectories
                ):
                    camera_color = self.camera_trajectories_state.camera_colors[i]
                    gl.glColor3f(camera_color[0], camera_color[1], camera_color[2])
                    glutils.DrawCameras(camera_trajectory, self.scale)
                for i, trajectory_line in enumerate(
                    self.camera_trajectories_state.camera_trajectory_lines
                ):
                    camera_color = self.camera_trajectories_state.camera_colors[i]
                    gl.glColor3f(camera_color[0], camera_color[1], camera_color[2])
                    glutils.DrawTrajectory(trajectory_line)

        # ==============================

        pangolin.FinishFrame()

    # draw sparse map (public interface - non-blocking, uses thread)
    def draw_slam_map(self, slam: "Slam"):
        """Request drawing of SLAM map (non-blocking, uses background thread)"""
        if self.qmap is None:
            return

        map: Map = slam.map
        viewer_cur_frame_data = ViewerCurrentFrameData()

        # We need to crete a snapshot of current frame data in the main slam thread, here, to avoid race conditions
        # or grabbing the frame current pose when the tracking is updating it.
        # On the other hand, we can grab a map snapshot in the drawer thread or main slam thread.
        f_cur_ref = slam.tracking.f_cur
        last_f = map.get_frame(-1)
        if last_f is not None:
            viewer_cur_frame_data.cur_frame_id = (
                f_cur_ref.id if f_cur_ref is not None else last_f.id
            )
            viewer_cur_frame_data.cur_pose = last_f.Twc()
            viewer_cur_frame_data.cur_pose_timestamp = last_f.timestamp
        elif f_cur_ref is not None:
            # Fallback to f_cur only if no frames in map yet (during initialization)
            viewer_cur_frame_data.cur_frame_id = f_cur_ref.id
            viewer_cur_frame_data.cur_pose = f_cur_ref.Twc()
            viewer_cur_frame_data.cur_pose_timestamp = f_cur_ref.timestamp
        else:
            viewer_cur_frame_data.cur_frame_id = -1

        if slam.tracking.predicted_pose is not None:
            viewer_cur_frame_data.predicted_pose = (
                slam.tracking.predicted_pose.inverse().matrix().copy()
            )

        if False:
            if slam.tracking.kf_ref is not None:
                viewer_cur_frame_data.reference_pose = slam.tracking.kf_ref.Twc()

        if self.slam_drawer_thread is not None:
            task = SlamDrawingTask(SlamDrawingTaskType.DRAW_SLAM_MAP, slam, viewer_cur_frame_data)
            self.slam_drawer_thread.request_draw(task)
        else:
            self._draw_slam_map_impl(slam, viewer_cur_frame_data)

    # draw dense map (public interface - non-blocking, uses thread)
    def draw_dense_map(self, slam: "Slam"):
        """Request drawing of dense map (non-blocking, uses background thread)"""
        if self.qdense is None:
            return
        if self.slam_drawer_thread is not None:
            task = SlamDrawingTask(SlamDrawingTaskType.DRAW_DENSE_MAP, slam)
            self.slam_drawer_thread.request_draw(task)
        else:
            self._draw_dense_map_impl(slam)

    # Internal implementation methods (called by the drawer thread)
    def _draw_slam_map_impl(self, slam: "Slam", viewer_cur_frame_data: ViewerCurrentFrameData):
        """Internal implementation of draw_slam_map (called by drawer thread)"""
        if self.qmap is None:
            return

        viewer_map_state = Viewer3DMapInput()
        viewer_map_state.cur_frame_data = viewer_cur_frame_data

        # get the map state data in the form of a set of data arrays for the viewer
        viewer_map_state.map_data = slam.map.get_data_arrays_for_drawing(
            max_points_to_visualize=int(kMaxSparseMapPointsToVisualize),
            min_weight_for_drawing_covisibility_edge=int(kMinWeightForDrawingCovisibilityEdge),
        )

        # Ground truth one-shot set
        if self.gt_trajectory is not None and not self._is_gt_set.value:
            viewer_map_state.gt_trajectory = np.asarray(self.gt_trajectory, dtype=np.float64)
            viewer_map_state.gt_timestamps = np.asarray(self.gt_timestamps, dtype=np.float64)
            viewer_map_state.align_gt_with_scale = self.align_gt_with_scale

        self.qmap.put(viewer_map_state)

    def _draw_dense_map_impl(self, slam: "Slam"):
        """Internal implementation of draw_dense_map (called by drawer thread)"""
        if self.qdense is None:
            return
        dense_map_output = slam.get_dense_map()
        if dense_map_output is not None:
            self.draw_dense_geometry(dense_map_output.point_cloud, dense_map_output.mesh)

    # inputs:
    #   point_cloud: VolumetricIntegrationPointCloud (see the file volumetric_integrator.py)
    #   mesh: VolumetricIntegrationMesh (see the file volumetric_integrator.py)
    #   camera_images: list of VizCameraImage objects
    def draw_dense_geometry(self, point_cloud=None, mesh=None, camera_images=None):
        if self.qdense is None:
            return
        if camera_images is None:
            camera_images = []
        dense_state = Viewer3DDenseInput()
        if mesh is not None:
            dense_state.mesh = (
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
                np.asarray(mesh.vertex_colors),
            )  # ,np.asarray(mesh.vertex_normals))
        else:
            if point_cloud is not None:
                points = np.asarray(point_cloud.points)
                colors = np.asarray(point_cloud.colors)
                if colors.shape[1] == 4:
                    colors = colors[:, 0:3]
                print(
                    f"Viewer3D: draw_dense_geometry - points.shape: {points.shape}, colors.shape: {colors.shape}"
                )
                semantic_colors = None
                if hasattr(point_cloud, "semantic_colors"):
                    semantic_colors = (
                        np.asarray(point_cloud.semantic_colors)
                        if point_cloud.semantic_colors is not None
                        else None
                    )
                instance_colors = None
                if hasattr(point_cloud, "instance_colors"):
                    instance_colors = (
                        np.asarray(point_cloud.instance_colors)
                        if point_cloud.instance_colors is not None
                        else None
                    )
                print(
                    f"Viewer3D: draw_dense_geometry - points.shape: {points.shape}, colors.shape: {colors.shape}"
                )
                if semantic_colors is not None and semantic_colors.shape[0] > 0:
                    print(
                        f"Viewer3D: draw_dense_geometry - semantic_colors.shape: {semantic_colors.shape}"
                    )
                if self.is_draw_instance_colors() and instance_colors is not None:
                    # overwrite the semantic colors with the instance colors
                    # (to avoid sending both semantic and instance colors to the viewer and we only need one)
                    semantic_colors = instance_colors
                dense_state.point_cloud = (points, colors, semantic_colors)
            else:
                Printer.orange("WARNING: both point_cloud and mesh are None")

        dense_state.camera_images = camera_images

        self.qdense.put(dense_state)

    def draw_map(self, map_state: Viewer3DMapInput):
        if self.qmap is None:
            return

        if self.gt_trajectory is not None:
            if not self._is_gt_set.value:
                map_state.gt_trajectory = np.asarray(self.gt_trajectory, dtype=np.float64)
                map_state.gt_timestamps = np.asarray(self.gt_timestamps, dtype=np.float64)
                map_state.align_gt_with_scale = self.align_gt_with_scale

        self.qmap.put(map_state)

    # inputs:
    #   camera_trajectories: list of arrays of camera poses (np 4x4 matrices Twc), each arrays is a distinct camera 6D trajectory
    #   camera_colors: list of colors (one for each camera array, not one for each camera)
    def draw_cameras(self, camera_trajectories, camera_colors=None, show_trajectory_line=False):
        if self.qcams is None:
            return
        camera_state = Viewer3DCameraTrajectoriesInput(show_trajectory_line)
        if camera_colors is None:
            for ca in camera_trajectories:
                camera_state.add(ca)
        else:
            assert len(camera_trajectories) == len(camera_colors)
            for ca, cc in zip(camera_trajectories, camera_colors):
                camera_state.add(ca, cc)

        self.qcams.put(camera_state)

    def draw_vo(self, vo):
        if self.qvo is None:
            return
        vo_state = Viewer3DVoInput()
        vo_state.poses = np.asarray(vo.poses)
        vo_state.pose_timestamps = np.asarray(vo.pose_timestamps, dtype=np.float64)
        vo_state.traj3d_est = np.asarray(vo.traj3d_est).reshape(-1, 3)
        vo_state.traj3d_gt = np.asarray(vo.traj3d_gt).reshape(-1, 3)

        self.qvo.put(vo_state)

    def updateTwc(self, pose):
        self.Twc.m = pose

    @staticmethod
    def drawPlane(num_divs=200, div_size=10, scale=1.0):
        # min_width = gl.glGetFloatv(gl.GL_ALIASED_LINE_WIDTH_RANGE)[0]
        # max_width = gl.glGetFloatv(gl.GL_ALIASED_LINE_WIDTH_RANGE)[1]
        # print(f"Line width supported range: {min_width} - {max_width}")
        gl.glLineWidth(0.1)
        # Plane parallel to x-z at origin with normal -y
        div_size = scale * div_size
        minx = -num_divs * div_size
        minz = -num_divs * div_size
        maxx = num_divs * div_size
        maxz = num_divs * div_size
        # gl.glLineWidth(2)
        # gl.glColor3f(0.7,0.7,1.0)
        gl.glColor3f(0.7, 0.7, 0.7)
        gl.glBegin(gl.GL_LINES)
        for n in range(2 * num_divs):
            gl.glVertex3f(minx + div_size * n, 0, minz)
            gl.glVertex3f(minx + div_size * n, 0, maxz)
            gl.glVertex3f(minx, 0, minz + div_size * n)
            gl.glVertex3f(maxx, 0, minz + div_size * n)
        gl.glEnd()
        gl.glLineWidth(1)
