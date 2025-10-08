import threading
import queue
from typing import TYPE_CHECKING, Dict, Tuple
import time
import subprocess
from pathlib import Path
import os
import gc
from functools import partial

import cv2
import numpy
import torch
import pyvirtualcam

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap
from app.processors.workers.frame_worker import FrameWorker
from app.processors.utils import faceutil
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import preset_actions

from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import layout_actions
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

class VideoProcessor(QObject):
    frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    webcam_frame_processed_signal = Signal(QPixmap, numpy.ndarray)
    single_frame_processed_signal = Signal(int, QPixmap, numpy.ndarray)
    def __init__(self, main_window: 'MainWindow', num_threads=2):
        super().__init__()
        self.main_window = main_window
        self.frame_queue = queue.Queue(maxsize=num_threads)
        self.media_capture: cv2.VideoCapture|None = None
        self.file_type = None
        self.fps = 0
        self.processing = False
        self.current_frame_number = 0
        self.max_frame_number = 0
        self.media_path = None
        self.num_threads = num_threads
        self.threads: Dict[int, threading.Thread] = {}

        self.current_frame: numpy.ndarray = []
        self.recording = False

        self.virtcam: pyvirtualcam.Camera|None = None

        self.recording_sp: subprocess.Popen|None = None 
        self.temp_file = '' 
        #Used to calculate the total processing time
        self.start_time = 0.0
        self.end_time = 0.0

        #Used to store the video start and enc seek time
        self.play_start_time = 0.0
        self.play_end_time = 0.0

        # Timer to manage frame reading intervals
        self.frame_read_timer = QTimer()
        self.frame_read_timer.timeout.connect(self.process_next_frame)

        self.next_frame_to_display = 0
        self.frame_processed_signal.connect(self.store_frame_to_display)
        self.frame_display_timer = QTimer()
        self.frame_display_timer.timeout.connect(self.display_next_frame)
        self.frames_to_display: Dict[int, Tuple[QPixmap, numpy.ndarray]] = {}


        self.webcam_frame_processed_signal.connect(self.store_webcam_frame_to_display)
        self.webcam_frames_to_display = queue.Queue()

        # Timer to update the gpu memory usage progressbar 
        self.gpu_memory_update_timer = QTimer()
        self.gpu_memory_update_timer.timeout.connect(partial(common_widget_actions.update_gpu_memory_progressbar, main_window))
        # AI Agent heartbeat on UI thread
        self.ai_agent_timer = QTimer()
        self.ai_agent_timer.timeout.connect(self._ai_agent_tick)

        self.single_frame_processed_signal.connect(self.display_current_frame)

        # --- Per-face temporal smoothing state for rotations ---
        # OneEuro filters keyed by target face_id
        self.rotation_filters = {}  # type: dict
        # Hysteresis engaged flags per face_id
        self.rotation_hysteresis = {}  # type: dict
        # Guard concurrent access from worker threads
        self.rotation_lock = threading.Lock()

        # --- Per-face temporal smoothing for transform (translation/scale) ---
        # Dict of face_id -> {'tx': OneEuroFilter, 'ty': OneEuroFilter, 's': OneEuroFilter}
        self.transform_filters = {}
        # Track last-seen timestamps per face for cleanup
        self._filter_last_seen = {}
        # Cleanup cadence: frames between sweeps and stale timeout (seconds)
        self._filter_cleanup_every_n_frames = 120
        self._filter_stale_timeout_sec = 30.0

        # --- Detection fallback (use previous frame) ---
        # Cache last successful detections and input frame to enable LK optical flow tracking when a frame misses detections
        self.last_detections = None  # dict with keys: bboxes, kpss_5, kpss, frame_number
        self.last_input_frame_rgb = None  # numpy RGB frame corresponding to last_detections
        self.last_input_frame_number = -1
        self.fallback_track_frames = 0  # length of current LK tracking chain
        self.fallback_max_chain = 3     # max consecutive frames to allow tracking without re-detection

        # --- Adaptive recognition threshold assist (per target face) ---
        # face_id -> consecutive frames with detected face but under-threshold similarity
        self.recog_no_match_streak = {}
        # face_id -> current effective threshold offset applied (negative values lower the threshold)
        self.recog_threshold_offset = {}

        # --- Scene metrics (updated every processed frame by FrameWorker) ---
        # Used by the AI Preset Agent to make decisions on the main/UI thread
        self.scene_metrics = {
            'motion': 0.0,
            'light': 0.5,
            'contrast': 0.5,
            'face_ratio': 0.0,
            'difficulty': 0.0,
        }
        # Agent state: currently active auto profile and last switch timestamp (monotonic seconds)
        self.ai_agent_state = {
            'active_profile': None,
            'last_switch_ts': 0.0,
        }
        # After a frame with no detections, briefly relax spatial lock to re-acquire
        self.relax_lock_frames = 0
        # Count consecutive frames with zero detections to adapt relaxation a bit
        self.det_no_face_streak = 0

    Slot(int, QPixmap, numpy.ndarray)
    def store_frame_to_display(self, frame_number, pixmap, frame):
        # print("Called store_frame_to_display()")
        self.frames_to_display[frame_number] = (pixmap, frame)

    # Use a queue to store the webcam frames, since the order of frames is not that important (Unless there are too many threads)
    Slot(QPixmap, numpy.ndarray)
    def store_webcam_frame_to_display(self, pixmap, frame):
        # print("Called store_webcam_frame_to_display()")
        self.webcam_frames_to_display.put((pixmap, frame))

    Slot(int, QPixmap, numpy.ndarray)
    def display_current_frame(self, frame_number, pixmap, frame):
        if self.main_window.loading_new_media:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number, reset_fit=True)
            self.main_window.loading_new_media = False

        else:
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, frame_number,)
        self.current_frame = frame
        torch.cuda.empty_cache()
        #Set GPU Memory Progressbar
        common_widget_actions.update_gpu_memory_progressbar(self.main_window)
        # Decay relax window for lock gating after a miss
        try:
            if self.relax_lock_frames > 0:
                self.relax_lock_frames -= 1
        except Exception:
            pass
    def display_next_frame(self):
        if not self.processing or (self.next_frame_to_display > self.max_frame_number):
            self.stop_processing()
        if self.next_frame_to_display not in self.frames_to_display:
            return
        else:
            pixmap, frame = self.frames_to_display.pop(self.next_frame_to_display)
            self.current_frame = frame

            # Check and send the frame to virtualcam, if the option is selected
            self.send_frame_to_virtualcam(frame)

            if self.recording:
                self.recording_sp.stdin.write(frame.tobytes())
            # Update the widget values using parameters if it is not recording (The updation of actual parameters is already done inside the FrameWorker, this step is to make the changes appear in the widgets)
            if not self.recording:
                video_control_actions.update_widget_values_from_markers(self.main_window, self.next_frame_to_display)
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, self.next_frame_to_display)
            self.threads.pop(self.next_frame_to_display)
            self.next_frame_to_display += 1

            # Periodic cleanup of stale per-face filters
            try:
                if (self.next_frame_to_display % self._filter_cleanup_every_n_frames) == 0:
                    now = time.perf_counter()
                    stale = []
                    for face_id, ts in list(self._filter_last_seen.items()):
                        if (now - ts) > self._filter_stale_timeout_sec:
                            stale.append(face_id)
                    if stale:
                        with self.rotation_lock:
                            for fid in stale:
                                self.transform_filters.pop(fid, None)
                                self.rotation_filters.pop(fid, None)
                                self.rotation_hysteresis.pop(fid, None)
                                self._filter_last_seen.pop(fid, None)
            except Exception:
                pass

    def display_next_webcam_frame(self):
        # print("Called display_next_webcam_frame()")
        if not self.processing:
            self.stop_processing()
        if self.webcam_frames_to_display.empty():
            # print("No Webcam frame found to display")
            return
        else:
            pixmap, frame = self.webcam_frames_to_display.get()
            self.current_frame = frame
            self.send_frame_to_virtualcam(frame)
            graphics_view_actions.update_graphics_view(self.main_window, pixmap, 0)

    def _ai_agent_tick(self):
        """Runs on UI thread. Selects an appropriate preset based on scene metrics.
        Applies with min dwell (cooldown) to avoid flicker. No-ops when disabled or no faces set.
        """
        try:
            if not self.main_window.control.get('AIAgentEnableToggle', False):
                return
            # Require at least one target face to be meaningful
            if not self.main_window.target_faces:
                return
            sm = self.scene_metrics or {}
            motion = float(sm.get('motion', 0.0))
            light = float(sm.get('light', 0.5))
            contrast = float(sm.get('contrast', 0.5))
            face_ratio = float(sm.get('face_ratio', 0.0))
            difficulty = float(sm.get('difficulty', 0.0))

            # Score presets; higher score wins
            # Heuristics:
            # - Vlog Mobile for high motion or tiny faces
            # - Low-Light for low light or low contrast
            # - Portrait Stable otherwise
            score_vlog = (motion * 2.0) + max(0.0, 0.15 - face_ratio) * 8.0
            score_low = max(0.0, 0.5 - light) * 3.0 + max(0.0, 0.5 - contrast) * 2.5
            score_portrait = 1.0 + max(0.0, face_ratio - 0.20) * 2.0 + max(0.0, 0.7 - motion) * 1.0

            # Pick best
            best = 'Portrait Stable'
            best_score = score_portrait
            if score_vlog > best_score:
                best = 'Vlog Mobile'; best_score = score_vlog
            if score_low > best_score:
                best = 'Low-Light'; best_score = score_low

            # Enforce min dwell
            min_dwell_s = float(self.main_window.control.get('AIAgentMinDwellSecondsSlider', 6))
            now = time.perf_counter()
            active = self.ai_agent_state.get('active_profile')
            last_ts = float(self.ai_agent_state.get('last_switch_ts', 0.0))
            if active == best and (now - last_ts) < min_dwell_s:
                return
            if (now - last_ts) < min_dwell_s:
                # Still within dwell; but allow upgrade to Vlog if motion is extremely high
                if best != active and best == 'Vlog Mobile' and motion > 1.2:
                    pass
                else:
                    return

            # Apply preset via existing action; this updates control and parameters safely
            preset_actions.apply_preset(self.main_window, best)
            self.ai_agent_state['active_profile'] = best
            self.ai_agent_state['last_switch_ts'] = now

            # Optional toast feedback
            try:
                common_widget_actions.create_and_show_toast_message(self.main_window, 'AI Agent', f'Profil appliqué: {best}', style_type='information')
            except Exception:
                pass
        except Exception:
            pass

    def send_frame_to_virtualcam(self, frame: numpy.ndarray):
        if self.main_window.control['SendVirtCamFramesEnableToggle'] and self.virtcam:
            # Check if the dimensions of the frame matches that of the Virtcam object
            # If it doesn't match, reinstantiate the Virtcam object with new dimensions
            height, width, _ = frame.shape
            if self.virtcam.height!=height or self.virtcam.width!=width:
                self.enable_virtualcam()
            try:
                self.virtcam.send(frame)
                self.virtcam.sleep_until_next_frame()
            except Exception as e:
                print(e)

    def set_number_of_threads(self, value):
        self.stop_processing()
        self.main_window.models_processor.set_number_of_threads(value)
        self.num_threads = value
        self.frame_queue = queue.Queue(maxsize=self.num_threads)
        print(f"Max Threads set as {value} ")

    def process_video(self):
        """Start video processing by reading frames and enqueueing them."""
        if self.processing:
            print("Processing already in progress. Ignoring start request.")
            return
            
        # Re-initialize the timers
        self.frame_display_timer = QTimer()
        self.frame_read_timer = QTimer()

        if self.file_type == 'video':
            self.frame_display_timer.timeout.connect(self.display_next_frame)
            self.frame_read_timer.timeout.connect(self.process_next_frame)

            if self.media_capture and self.media_capture.isOpened():
                print("Starting video processing.")
                if self.recording:
                    layout_actions.disable_all_parameters_and_control_widget(self.main_window)

                self.start_time = time.perf_counter()
                self.processing = True
                self.frames_to_display.clear()
                self.threads.clear()

                if self.recording:
                    self.create_ffmpeg_subprocess()

                self.play_start_time = float(self.media_capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))

                if self.main_window.control['VideoPlaybackCustomFpsToggle']:
                    fps = self.main_window.control['VideoPlaybackCustomFpsSlider']
                else:
                    fps = self.media_capture.get(cv2.CAP_PROP_FPS)
                
                interval = 1000 / fps if fps > 0 else 30
                interval = int(interval * 0.8) #Process 20% faster to offset the frame loading & processing time so the video will be played close to the original fps
                print(f"Starting frame_read_timer with an interval of {interval} ms.")
                if self.recording:
                    self.frame_read_timer.start()
                    self.frame_display_timer.start()
                else:
                    self.frame_read_timer.start(interval)
                    self.frame_display_timer.start()
                self.gpu_memory_update_timer.start(5000) #Update GPU memory progressbar every 5 Seconds
                # Start AI Agent timer; tick will no-op if disabled
                try:
                    self.ai_agent_timer.start(700)  # ~1.4 Hz
                except Exception:
                    pass

            else:
                print("Error: Unable to open the video.")
                self.processing = False
                self.frame_read_timer.stop()
                video_control_actions.set_play_button_icon_to_play(self.main_window)
                try:
                    self.ai_agent_timer.stop()
                except Exception:
                    pass
        # 
        elif self.file_type == 'webcam':
            print("Calling process_video() on Webcam stream")
            self.processing = True
            self.frames_to_display.clear()
            self.threads.clear()
            fps = self.media_capture.get(cv2.CAP_PROP_FPS)
            interval = 1000 / fps if fps > 0 else 30
            interval = int(interval * 0.8) #Process 20% faster to offset the frame loading & processing time so the video will be played close to the original fps
            self.frame_read_timer.timeout.connect(self.process_next_webcam_frame)
            self.frame_read_timer.start(interval)
            self.frame_display_timer.timeout.connect(self.display_next_webcam_frame)
            self.frame_display_timer.start()
            self.gpu_memory_update_timer.start(5000) #Update GPU memory progressbar every 5 Seconds
            # Start AI Agent timer; tick will no-op if disabled
            try:
                self.ai_agent_timer.start(700)
            except Exception:
                pass



    def process_next_frame(self):
        """Read the next frame and add it to the queue for processing."""

        if self.current_frame_number > self.max_frame_number:
            # print("Stopping frame_read_timer as all frames have been read!")
            self.frame_read_timer.stop()
            return

        if self.frame_queue.qsize() >= self.num_threads:
            # print(f"Queue is full ({self.frame_queue.qsize()} frames). Throttling frame reading.")
            return

        if self.file_type == 'video' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode = not self.recording)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame)
                self.current_frame_number += 1
            else:
                print("Cannot read frame!", self.current_frame_number)
                self.stop_processing()
                self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {self.current_frame_number}.\n Stopped Processing...!', self.main_window)

    def start_frame_worker(self, frame_number, frame, is_single_frame=False):
        """Start a FrameWorker to process the given frame."""
        worker = FrameWorker(frame, self.main_window, frame_number, self.frame_queue, is_single_frame)
        self.threads[frame_number] = worker
        if is_single_frame:
            worker.run()
        else:
            worker.start()

    def process_current_frame(self):

        # print("\nCalled process_current_frame()",self.current_frame_number)
        # self.main_window.processed_frames.clear()

        self.next_frame_to_display = self.current_frame_number
        if self.file_type == 'video' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode=False)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
                
                self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            else:
                print("Cannot read frame!", self.current_frame_number)
                self.main_window.display_messagebox_signal.emit('Error Reading Frame', f'Error Reading Frame {self.current_frame_number}.', self.main_window)

        # """Process a single image frame directly without queuing."""
        elif self.file_type == 'image':
            frame = misc_helpers.read_image_file(self.media_path)
            if frame is not None:

                frame = frame[..., ::-1]  # Convert BGR to RGB
                self.frame_queue.put(self.current_frame_number)
                # print("Processing current frame as image.")
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
            else:
                print("Error: Unable to read image file.")

        # Handle webcam capture
        elif self.file_type == 'webcam':
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode = False)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame, is_single_frame=True)
            else:
                print("Unable to read Webcam frame!")
        self.join_and_clear_threads()

    def process_next_webcam_frame(self):
        # print("Called process_next_webcam_frame()")

        if self.frame_queue.qsize() >= self.num_threads:
            # print(f"Queue is full ({self.frame_queue.qsize()} frames). Throttling frame reading.")
            return
        if self.file_type == 'webcam' and self.media_capture:
            ret, frame = misc_helpers.read_frame(self.media_capture, preview_mode = False)
            if ret:
                frame = frame[..., ::-1]  # Convert BGR to RGB
                # print(f"Enqueuing frame {self.current_frame_number}")
                self.frame_queue.put(self.current_frame_number)
                self.start_frame_worker(self.current_frame_number, frame)

    # @misc_helpers.benchmark
    def stop_processing(self):
        """Stop video processing and signal completion."""
        if not self.processing:
            # print("Processing not active. No action to perform.")
            video_control_actions.reset_media_buttons(self.main_window)

            return False
        
        print("Stopping video processing.")
        self.processing = False
        
        if self.file_type=='video' or self.file_type=='webcam':

            # print("Stopping Timers")
            self.frame_read_timer.stop()
            self.frame_display_timer.stop()
            self.gpu_memory_update_timer.stop()
            try:
                self.ai_agent_timer.stop()
            except Exception:
                pass
            self.join_and_clear_threads()


            # print("Clearing Threads and Queues")
            self.threads.clear()
            self.frames_to_display.clear()
            self.webcam_frames_to_display.queue.clear()

            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()

            self.current_frame_number = self.main_window.videoSeekSlider.value()
            self.media_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)

            if self.recording and self.file_type=='video':
                self.recording_sp.stdin.close()
                self.recording_sp.wait()

            self.play_end_time = float(self.media_capture.get(cv2.CAP_PROP_POS_FRAMES) / float(self.fps))

            if self.file_type=='video':
                if self.recording:
                    final_file_path = misc_helpers.get_output_file_path(self.media_path, self.main_window.control['OutputMediaFolder'])
                    if Path(final_file_path).is_file():
                        os.remove(final_file_path)
                    print("Adding audio...")
                    args = ["ffmpeg",
                            '-hide_banner',
                            '-loglevel',    'error',
                            "-i", self.temp_file,
                            "-ss", str(self.play_start_time), "-to", str(self.play_end_time), "-i",  self.media_path,
                            "-c",  "copy", # may be c:v
                            "-map", "0:v:0", "-map", "1:a:0?",
                            "-shortest",
                            final_file_path]
                    subprocess.run(args, check=False) #Add Audio
                    os.remove(self.temp_file)

                self.end_time = time.perf_counter()
                processing_time = self.end_time - self.start_time
                print(f"\nProcessing completed in {processing_time} seconds")
                avg_fps = ((self.play_end_time - self.play_start_time) * self.fps) / processing_time
                print(f'Average FPS: {avg_fps}\n')

                if self.recording:
                    layout_actions.enable_all_parameters_and_control_widget(self.main_window)

            self.recording = False #Set recording as False to make sure the next process_video() call doesnt not record the video, unless the user press the record button

            print("Clearing Cache")
            torch.cuda.empty_cache()
            gc.collect()
            video_control_actions.reset_media_buttons(self.main_window)
            print("Successfully Stopped Processing")
            return True
        
    def join_and_clear_threads(self):
        # print("Joining Threads")
        for _, thread in self.threads.items():
            if thread.is_alive():
                thread.join()
        # print('Clearing Threads')
        self.threads.clear()
    
    def create_ffmpeg_subprocess(self):
        # Use Dimensions of the last processed frame as it could be different from the original frame due to restorers and frame enhancers 
        frame_height, frame_width, _ = self.current_frame.shape

        self.temp_file = r'temp_output.mp4'
        if Path(self.temp_file).is_file():
            os.remove(self.temp_file)

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",             # Specify raw video input
            "-pix_fmt", "bgr24",          # Pixel format of input frames
            "-s", f"{frame_width}x{frame_height}",  # Frame resolution
            "-r", str(self.fps),          # Frame rate
            "-i", "pipe:",                # Input from stdin
            "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",  # Padding and format conversion            
            "-c:v", "libx264",            # H.264 codec
            "-crf", "18",                 # Quality setting
            self.temp_file                # Output file
        ]

        self.recording_sp = subprocess.Popen(args, stdin=subprocess.PIPE)

    def enable_virtualcam(self, backend=False):
        #Check if capture contains any cv2 stream or is it an empty list
        if self.media_capture:
            if isinstance(self.current_frame, numpy.ndarray):
                frame_height, frame_width, _ = self.current_frame.shape
            else:
                frame_height = int(self.media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_width = int(self.media_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.disable_virtualcam()
            try:
                backend = backend or self.main_window.control['VirtCamBackendSelection']
                # self.virtcam = pyvirtualcam.Camera(width=vid_width, height=vid_height, fps=int(self.fps), backend='unitycapture', device='Unity Video Capture')
                self.virtcam = pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=int(self.fps), backend=backend, fmt=pyvirtualcam.PixelFormat.BGR)

            except Exception as e:
                print(e)

    def disable_virtualcam(self):
        if self.virtcam:
            self.virtcam.close()
        self.virtcam = None