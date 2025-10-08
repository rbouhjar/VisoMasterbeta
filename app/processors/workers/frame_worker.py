import traceback
from typing import TYPE_CHECKING
import threading
from math import floor, ceil

import torch
from skimage import transform as trans

from torchvision.transforms import v2
import torchvision
from torchvision import transforms

import numpy as np
import cv2

from app.processors.utils import faceutil
from app.helpers.logger import log_detection_gap, log_swap_gap
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.helpers.miscellaneous import t512,t384,t256,t128, ParametersDict

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

torchvision.disable_beta_transforms_warning()

class FrameWorker(threading.Thread):
    def __init__(self, frame, main_window: 'MainWindow', frame_number, frame_queue, is_single_frame=False):
        super().__init__()
        self.frame_queue = frame_queue
        self.frame = frame
        self.main_window = main_window
        self.frame_number = frame_number
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor
        self.is_single_frame = is_single_frame
        self.parameters = {}
        self.target_faces = main_window.target_faces
        self.compare_images = []
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False

    def run(self):
        try:
            # Update parameters from markers (if exists) without concurrent access from other threads
            with self.main_window.models_processor.model_lock:
                video_control_actions.update_parameters_and_control_from_marker(self.main_window, self.frame_number)
            self.parameters = self.main_window.parameters.copy()
            # Check if view mask or face compare checkboxes are checked
            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked() 
            self.is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked() 

            # Process the frame with model inference
            # print(f"Processing frame {self.frame_number}")
            if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked() or self.main_window.control['FrameEnhancerEnableToggle']:
                self.frame = self.process_frame()
            else:
                # Img must be in BGR format
                self.frame = self.frame[..., ::-1]  # Swap the channels from RGB to BGR
            self.frame = np.ascontiguousarray(self.frame)

            # Display the frame if processing is still active

            pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, self.frame)

            # Output processed Webcam frame
            if self.video_processor.file_type=='webcam' and not self.is_single_frame:
                self.video_processor.webcam_frame_processed_signal.emit(pixmap, self.frame)

            #Output Video frame (while playing)
            elif not self.is_single_frame:
                self.video_processor.frame_processed_signal.emit(self.frame_number, pixmap, self.frame)
            # Output Image/Video frame (Single frame)
            else:
                # print('Emitted single_frame_processed_signal')
                self.video_processor.single_frame_processed_signal.emit(self.frame_number, pixmap, self.frame)


            # Mark the frame as done in the queue
            self.video_processor.frame_queue.get()
            self.video_processor.frame_queue.task_done()

            # Check if playback is complete
            if self.video_processor.frame_queue.empty() and not self.video_processor.processing and self.video_processor.next_frame_to_display >= self.video_processor.max_frame_number:
                self.video_processor.stop_processing()

        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"Error in FrameWorker: {e}")
            traceback.print_exc()
    
    # @misc_helpers.benchmark
    def process_frame(self):
        # Load frame into VRAM
        curr_frame_rgb = self.frame  # numpy RGB copy for tracking
        img = torch.from_numpy(self.frame.astype('uint8')).to(self.models_processor.device) #HxWxc
        img = img.permute(2,0,1)#cxHxW

        #Scale up frame if it is smaller than 512
        img_x = img.size()[2]
        img_y = img.size()[1]

        # det_scale = 1.0
        if img_x<512 and img_y<512:
            # if x is smaller, set x to 512
            if img_x <= img_y:
                new_height = int(512*img_y/img_x)
                tscale = v2.Resize((new_height, 512), antialias=True)
            else:
                new_height = 512
                tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=True)

            img = tscale(img)

            # det_scale = torch.div(new_height, img_y)

        elif img_x<512:
            new_height = int(512*img_y/img_x)
            tscale = v2.Resize((new_height, 512), antialias=True)
            img = tscale(img)

            # det_scale = torch.div(new_height, img_y)

        elif img_y<512:
            new_height = 512
            tscale = v2.Resize((new_height, int(512*img_x/img_y)), antialias=True)
            img = tscale(img)

            # det_scale = torch.div(new_height, img_y)

        control = self.main_window.control.copy()
        # Defaults for AI Smart Tuning fallbacks
        ai_tune_on = False
        ai_thr_ease = 0.0
        ai_feather_pct = 0.0
        ai_edge_strength = 0.0
        # Rotate the frame
        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        use_landmark_detection=control['LandmarkDetectToggle']
        landmark_detect_mode=control['LandmarkDetectModelSelection']
        from_points = control["DetectFromPointsToggle"]
        if self.main_window.editFacesButton.isChecked():
            if not use_landmark_detection or landmark_detect_mode=="5":
                # force to use landmark detector when edit face is enabled.
                use_landmark_detection = True
                landmark_detect_mode = "203"

            # force to use from_points in landmark detector when edit face is enabled.
            from_points = True

        bboxes, kpss_5, kpss = self.models_processor.run_detect(
            img,
            control['DetectorModelSelection'],
            max_num=control['MaxFacesToDetectSlider'],
            score=control['DetectorScoreSlider']/100.0,
            input_size=(512, 512),
            use_landmark_detection=use_landmark_detection,
            landmark_detect_mode=landmark_detect_mode,
            landmark_score=control["LandmarkDetectScoreSlider"]/100.0,
            from_points=from_points,
            rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
        )

        # Soft retry: if no detection, try once more with a slightly lower threshold on the full frame
        soft_retry_done = False
        if (len(kpss_5) == 0 or len(bboxes) == 0):
            try:
                det_score = control['DetectorScoreSlider'] / 100.0
                redet_score = max(0.15, det_score - 0.1)
                soft_retry_done = True
                b2, k52, k2 = self.models_processor.run_detect(
                    img,
                    control['DetectorModelSelection'],
                    max_num=control['MaxFacesToDetectSlider'],
                    score=redet_score,
                    input_size=(512, 512),
                    use_landmark_detection=use_landmark_detection,
                    landmark_detect_mode=landmark_detect_mode,
                    landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
                    from_points=from_points,
                    rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
                )
                if len(k52) > 0 and len(b2) > 0:
                    bboxes, kpss_5, kpss = b2, k52, k2
            except Exception:
                pass

        # Fallback: if no detections, try tracking last landmarks from previous frame via LK optical flow
        used_tracking = False
        if (len(kpss_5) == 0 or len(bboxes) == 0):
            last = getattr(self.video_processor, 'last_detections', None)
            # Read UI fallback config
            max_chain = int(control.get('DetectionFallbackMaxFramesSlider', getattr(self.video_processor, 'fallback_max_chain', 3)))
            roi_toggle = bool(control.get('DetectionRedetectROIToggle', True))
            roi_pad_pct = float(control.get('DetectionRedetectROIPaddingSlider', 35))
            if (
                isinstance(last, dict)
                and self.video_processor.last_input_frame_rgb is not None
                and (self.frame_number - last.get('frame_number', -1)) == 1
                and getattr(self.video_processor, 'fallback_track_frames', 0) < max_chain
            ):
                tracked = None
                try:
                    tracked = self._track_kps_lk(self.video_processor.last_input_frame_rgb, curr_frame_rgb, last['kpss_5'], last['bboxes'])
                except Exception:
                    tracked = None
                if tracked is not None and tracked['kpss_5'].shape[0] > 0:
                    bboxes = tracked['bboxes']
                    kpss_5 = tracked['kpss_5']
                    # Transform dense landmarks if available
                    if 'kpss' in last and isinstance(last['kpss'], np.ndarray) and last['kpss'].ndim == 3 and last['kpss'].shape[0] == kpss_5.shape[0]:
                        kpss = self._transform_all_landmarks(last['kpss'], last['kpss_5'], kpss_5)
                    used_tracking = True

                    # Optional ROI re-detect to break tracking chains
                    if roi_toggle and len(bboxes) > 0:
                        try:
                            x0 = int(np.floor(np.min(bboxes[:, 0])));
                            y0 = int(np.floor(np.min(bboxes[:, 1])));
                            x1 = int(np.ceil(np.max(bboxes[:, 2])));
                            y1 = int(np.ceil(np.max(bboxes[:, 3])));
                            pad = max(0.0, roi_pad_pct) / 100.0
                            w = x1 - x0; h = y1 - y0
                            cx = x0 + w / 2.0; cy = y0 + h / 2.0
                            w2 = int(w * (1 + pad)); h2 = int(h * (1 + pad))
                            rx0 = max(0, int(cx - w2 / 2)); ry0 = max(0, int(cy - h2 / 2))
                            rx1 = min(curr_frame_rgb.shape[1] - 1, int(cx + w2 / 2))
                            ry1 = min(curr_frame_rgb.shape[0] - 1, int(cy + h2 / 2))
                            if rx1 > rx0 + 10 and ry1 > ry0 + 10:
                                roi_rgb = curr_frame_rgb[ry0:ry1, rx0:rx1, :]
                                roi_t = torch.from_numpy(roi_rgb.astype('uint8')).to(self.models_processor.device).permute(2, 0, 1)
                                # Ensure min size ~512 like main path
                                r_h, r_w = roi_t.size()[1], roi_t.size()[2]
                                if r_w < 512 or r_h < 512:
                                    if r_w <= r_h:
                                        new_h = int(512 * r_h / r_w)
                                        tscale = v2.Resize((new_h, 512), antialias=True)
                                    else:
                                        new_h = 512
                                        tscale = v2.Resize((new_h, int(512 * r_w / r_h)), antialias=True)
                                    roi_t = tscale(roi_t)
                                # Slightly lower score to be sticky on ROI
                                det_score = control['DetectorScoreSlider'] / 100.0
                                redet_score = max(0.15, det_score - 0.1)
                                b_roi, k5_roi, k_roi = self.models_processor.run_detect(
                                    roi_t,
                                    control['DetectorModelSelection'],
                                    max_num=control['MaxFacesToDetectSlider'],
                                    score=redet_score,
                                    input_size=(512, 512),
                                    use_landmark_detection=use_landmark_detection,
                                    landmark_detect_mode=landmark_detect_mode,
                                    landmark_score=control["LandmarkDetectScoreSlider"] / 100.0,
                                    from_points=False,
                                    rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270]
                                )
                                if len(k5_roi) > 0 and len(b_roi) > 0:
                                    bboxes = b_roi.copy(); bboxes[:, 0] += rx0; bboxes[:, 1] += ry0; bboxes[:, 2] += rx0; bboxes[:, 3] += ry0
                                    kpss_5 = k5_roi.copy(); kpss_5[:, :, 0] += rx0; kpss_5[:, :, 1] += ry0
                                    if isinstance(k_roi, np.ndarray) and k_roi.ndim == 3:
                                        kpss = k_roi.copy(); kpss[:, :, 0] += rx0; kpss[:, :, 1] += ry0
                                    used_tracking = False  # we re-acquired; treat as fresh detection
                        except Exception:
                            pass
        # If still no detections after soft retry and any fallback attempts, log this frame number
        if (len(kpss_5) == 0 or len(bboxes) == 0):
            try:
                extras = {
                    'soft_retry': bool(soft_retry_done),
                    'fallback': bool(used_tracking),
                    'reason': 'no_detections_after_retries'
                }
                log_detection_gap(self.frame_number, extras)
                # Relax lock for the next couple of frames so we can re-acquire
                try:
                    st = int(getattr(self.video_processor, 'det_no_face_streak', 0)) + 1
                    self.video_processor.det_no_face_streak = st
                    # Scale relaxation softly: 2 + min(3, streak//2)
                    base = 2 + min(3, st // 2)
                    self.video_processor.relax_lock_frames = max(int(getattr(self.video_processor, 'relax_lock_frames', 0)), base)
                except Exception:
                    pass
                # Performance fast-path: if enabled, skip any further actions when no swap is possible
                try:
                    if bool(control.get('PerformanceSkipWhenNoSwapEnableToggle', True)):
                        # Return original frame (BGR)
                        return curr_frame_rgb[..., ::-1]
                except Exception:
                    pass
            except Exception:
                pass
        else:
            # We had detections; reset the miss streak
            try:
                self.video_processor.det_no_face_streak = 0
            except Exception:
                pass
        
        # Optional temporal smoothing of detections to reduce jitter
        if len(kpss_5) > 0 and len(bboxes) > 0 and bool(control.get('DetectionTemporalSmoothingEnableToggle', True)):
            try:
                last = getattr(self.video_processor, 'last_detections', None)
                if isinstance(last, dict) and last.get('frame_number', -999) >= 0 and isinstance(last.get('bboxes'), np.ndarray):
                    # Match current faces to previous by nearest center (fast heuristic)
                    prev_b = last['bboxes']
                    prev_k5 = last.get('kpss_5')
                    alpha = float(control.get('DetectionTemporalSmoothingStrengthSlider', 40)) / 100.0
                    alpha = max(0.0, min(alpha, 0.9))
                    if prev_b.shape[0] > 0:
                        # Build centers and pairwise distances
                        cur_centers = np.stack([(bboxes[:,0]+bboxes[:,2])*0.5, (bboxes[:,1]+bboxes[:,3])*0.5], axis=1)  # (N,2)
                        prev_centers = np.stack([(prev_b[:,0]+prev_b[:,2])*0.5, (prev_b[:,1]+prev_b[:,3])*0.5], axis=1)  # (M,2)
                        # Pairwise euclidean distances
                        d = np.sqrt(np.maximum(0.0, ((cur_centers[:,None,:]-prev_centers[None,:,:])**2).sum(axis=2)))  # (N,M)
                        # Normalization by average diagonal per pair
                        cur_diag = np.sqrt(np.maximum(0.0, (bboxes[:,2]-bboxes[:,0])**2 + (bboxes[:,3]-bboxes[:,1])**2))  # (N,)
                        prev_diag = np.sqrt(np.maximum(0.0, (prev_b[:,2]-prev_b[:,0])**2 + (prev_b[:,3]-prev_b[:,1])**2))  # (M,)
                        avg_diag = 0.5*(cur_diag[:,None] + prev_diag[None,:])
                        avg_diag = np.where(avg_diag <= 1e-6, 1.0, avg_diag)
                        cdist_norm = np.clip(d/avg_diag, 0.0, 2.0)  # ~0 close, >1 far

                        # Pairwise IoU
                        x1 = np.maximum(bboxes[:,None,0], prev_b[None,:,0])
                        y1 = np.maximum(bboxes[:,None,1], prev_b[None,:,1])
                        x2 = np.minimum(bboxes[:,None,2], prev_b[None,:,2])
                        y2 = np.minimum(bboxes[:,None,3], prev_b[None,:,3])
                        iw = np.maximum(0.0, x2 - x1)
                        ih = np.maximum(0.0, y2 - y1)
                        inter = iw*ih
                        area_c = np.maximum(0.0, (bboxes[:,2]-bboxes[:,0])) * np.maximum(0.0, (bboxes[:,3]-bboxes[:,1]))
                        area_p = np.maximum(0.0, (prev_b[:,2]-prev_b[:,0])) * np.maximum(0.0, (prev_b[:,3]-prev_b[:,1]))
                        union = area_c[:,None] + area_p[None,:] - inter
                        iou = np.where(union > 0, inter/union, 0.0)

                        # Combined cost: lower is better
                        cost = 0.5*cdist_norm + 0.5*(1.0 - iou)

                        # Mutual best matching with simple thresholds
                        cur_best = np.argmin(cost, axis=1)  # (N,)
                        prev_best = np.argmin(cost, axis=0)  # (M,)
                        used_prev = set()
                        for i in range(bboxes.shape[0]):
                            j = int(cur_best[i])
                            if j < 0 or j >= prev_b.shape[0] or j in used_prev:
                                continue
                            # mutual best and acceptable pairing
                            if prev_best[j] == i:
                                # Also ensure at least tiny overlap or proximity
                                if (iou[i,j] >= 0.05) or (cdist_norm[i,j] <= 0.6):
                                    used_prev.add(j)
                                    # EMA for bbox
                                    bboxes[i,0:4] = (1.0-alpha)*bboxes[i,0:4] + alpha*prev_b[j,0:4]
                                    # EMA for 5pt landmarks
                                    if isinstance(prev_k5, np.ndarray) and prev_k5.shape[0] > j:
                                        kpss_5[i] = (1.0-alpha)*kpss_5[i] + alpha*prev_k5[j]
                                    # EMA for dense landmarks if available now and before
                                    if isinstance(kpss, np.ndarray) and isinstance(last.get('kpss'), np.ndarray):
                                        if kpss.ndim == 3 and last['kpss'].ndim == 3 and last['kpss'].shape[0] > j and kpss.shape[1:] == last['kpss'].shape[1:]:
                                            kpss[i] = (1.0-alpha)*kpss[i] + alpha*last['kpss'][j]
                        # Greedy fallback for any unmatched current faces
                        if len(used_prev) < prev_b.shape[0]:
                            for i in range(bboxes.shape[0]):
                                if any(prev_best == i) and int(cur_best[i]) in used_prev:
                                    continue
                                j = int(cur_best[i])
                                if j >= 0 and j < prev_b.shape[0] and j not in used_prev:
                                    if (iou[i,j] >= 0.2) or (cdist_norm[i,j] <= 0.4):
                                        used_prev.add(j)
                                        bboxes[i,0:4] = (1.0-alpha)*bboxes[i,0:4] + alpha*prev_b[j,0:4]
                                        if isinstance(prev_k5, np.ndarray) and prev_k5.shape[0] > j:
                                            kpss_5[i] = (1.0-alpha)*kpss_5[i] + alpha*prev_k5[j]
                                        if isinstance(kpss, np.ndarray) and isinstance(last.get('kpss'), np.ndarray):
                                            if kpss.ndim == 3 and last['kpss'].ndim == 3 and last['kpss'].shape[0] > j and kpss.shape[1:] == last['kpss'].shape[1:]:
                                                kpss[i] = (1.0-alpha)*kpss[i] + alpha*last['kpss'][j]
            except Exception:
                pass

        # Update caches for next frame
        if len(kpss_5) > 0 and len(bboxes) > 0:
            self.video_processor.last_detections = {
                'bboxes': bboxes,
                'kpss_5': kpss_5.copy(),
                'kpss': kpss.copy() if isinstance(kpss, np.ndarray) else kpss,
                'frame_number': self.frame_number,
            }
            self.video_processor.last_input_frame_rgb = curr_frame_rgb
            self.video_processor.last_input_frame_number = self.frame_number
            if used_tracking:
                self.video_processor.fallback_track_frames = getattr(self.video_processor, 'fallback_track_frames', 0) + 1
            else:
                self.video_processor.fallback_track_frames = 0

        det_faces_data = []
        if len(kpss_5) > 0:
            # Build a safe dense-landmarks container: fall back to 5-point landmarks if detector didn't return dense points
            num_faces = int(kpss_5.shape[0])
            use_dense = isinstance(kpss, np.ndarray) and kpss.ndim == 3 and kpss.shape[0] >= num_faces
            for i in range(num_faces):
                face_kps_5 = kpss_5[i]
                if use_dense:
                    face_kps_all = kpss[i]
                else:
                    face_kps_all = face_kps_5  # safe fallback
                # Ensure numeric dtype for landmarks to avoid downstream linalg casting errors
                try:
                    fk5 = np.asarray(face_kps_5, dtype=np.float32)
                except Exception:
                    fk5 = face_kps_5
                face_emb, _ = self.models_processor.run_recognize_direct(img, fk5, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                det_faces_data.append({'kps_5': face_kps_5, 'kps_all': face_kps_all, 'embedding': face_emb, 'bbox': bboxes[i]})

        compare_mode = self.is_view_face_mask or self.is_view_face_compare

        # Counters and diagnostics for swap logging
        detected_count = len(det_faces_data)
        matched_count = 0
        attempted_count = 0
        applied_count = 0
        # Track top similarity and target when no match
        top_best_sim = -1.0
        top_best_target_id = None
        recog_model_name = str(control.get('RecognitionModelSelection', ''))

    # --- AI Smart Tuning (transient per-frame adjustments) ---
        ai_tune_on = bool(control.get('AISmartTuningEnableToggle', False))
        ai_mode = str(control.get('AISmartTuningModeSelection', 'Balanced'))
        ai_react = max(1.0, float(control.get('AISmartTuningReactivitySlider', 12)))
        # Mode multipliers
        if ai_mode == 'Conservative':
            ai_gain = 0.5
        elif ai_mode == 'Aggressive':
            ai_gain = 1.5
        else:
            ai_gain = 1.0
        # Precompute scene metrics for tuning
        try:
            # Motion score: average optical flow magnitude of a sparse grid (fast heuristic)
            motion_score = 0.0
            if isinstance(curr_frame_rgb, np.ndarray) and getattr(self.video_processor, 'last_input_frame_rgb', None) is not None:
                prev = self.video_processor.last_input_frame_rgb
                # Downscale for speed
                small_prev = cv2.resize(prev, (0,0), fx=0.25, fy=0.25)
                small_curr = cv2.resize(curr_frame_rgb, (0,0), fx=0.25, fy=0.25)
                flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(small_prev, cv2.COLOR_RGB2GRAY), cv2.cvtColor(small_curr, cv2.COLOR_RGB2GRAY), None, 0.5, 1, 10, 2, 5, 1.1, 0)
                mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                motion_score = float(np.clip(np.median(mag), 0.0, 5.0))  # 0..~5
        except Exception:
            motion_score = 0.0
        # Lightness/contrast heuristic from current frame
        try:
            gray = cv2.cvtColor(curr_frame_rgb, cv2.COLOR_RGB2GRAY)
            light = float(np.clip(np.mean(gray) / 255.0, 0.0, 1.0))
            contrast = float(np.clip(np.std(gray) / 64.0, 0.0, 1.0))
        except Exception:
            light = 0.5; contrast = 0.5
        # Face size heuristic (relative to frame height)
        try:
            face_ratio = 0.0
            if len(det_faces_data) > 0:
                h, w, _ = curr_frame_rgb.shape
                # take largest face bbox height
                fr = 0.0
                for d in det_faces_data:
                    y0 = float(d['bbox'][1]); y1 = float(d['bbox'][3])
                    fr = max(fr, (y1 - y0) / max(1.0, h))
                face_ratio = float(np.clip(fr, 0.0, 1.0))
        except Exception:
            face_ratio = 0.0
        # Pose heuristic: roll already computed per face later; here track if extreme pitch is likely (via AutoPitch estimator would refine)
        # We'll infer difficulty if face is small, motion high, or light low
        difficulty = 0.0
        try:
            difficulty = float(np.clip((1.0 - light) * 0.5 + (1.0 - contrast) * 0.3 + np.clip(motion_score / 2.0, 0.0, 1.0) * 0.2 + (0.15 - face_ratio) * 2.0, 0.0, 1.5))
        except Exception:
            difficulty = 0.0
        # Build transient adjustments (not saved): threshold ease and mask smoothing
        ai_thr_ease = 0.0
        ai_feather_pct = 0.0
        ai_edge_strength = 0.0
        if ai_tune_on:
            # Ease threshold up to 6 points in very hard conditions (scaled by mode)
            ai_thr_ease = min(6.0, 6.0 * difficulty * ai_gain)
            # Add feather and edge smoothing when motion/low light present
            ai_feather_pct = min(20.0, (8.0 * motion_score + (1.0 - light) * 12.0) * ai_gain)
            ai_edge_strength = min(12.0, (6.0 * motion_score + (1.0 - contrast) * 10.0) * ai_gain)

        # Share scene metrics with the main thread for AI Preset Agent decisions
        try:
            self.video_processor.scene_metrics.update({
                'motion': float(motion_score),
                'light': float(light),
                'contrast': float(contrast),
                'face_ratio': float(face_ratio),
                'difficulty': float(difficulty),
            })
        except Exception:
            pass
        
        if det_faces_data:
            # Loop through target faces to see if they match our found face embeddings
            for i, fface in enumerate(det_faces_data):
                    for _, target_face in self.main_window.target_faces.items():
                        # Optional: restrict detection to a tiny zone near the last swapped face for this target
                        # Do not hard-gate by lock here; always evaluate similarity.
                        # Lock will still be used for local easing, pixel force bypass, and selection preference.
                        try:
                            _ = bool(control.get('DetectionLockToLastSwappedEnableToggle', True))
                            _ = int(getattr(self.video_processor, 'relax_lock_frames', 0))
                            # Compute last center if needed (used later for easing/force); no skipping here.
                            _ = getattr(self.video_processor, 'last_swapped_center', {})
                        except Exception:
                            pass
                        # Safely initialize parameters for this face if missing
                        if target_face.face_id not in self.parameters:
                            try:
                                self.parameters[target_face.face_id] = self.main_window.default_parameters.copy()
                            except Exception:
                                self.parameters[target_face.face_id] = dict(self.main_window.default_parameters)
                        parameters = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters) #Use the parameters of the target face

                        if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked():
                            sim = self.models_processor.findCosineDistance(fface['embedding'], target_face.get_embedding(control['RecognitionModelSelection'])) # Recognition for comparing
                            # Adaptive threshold assist (optional)
                            eff_thr = float(parameters['SimilarityThresholdSlider'])
                            # Local easing when detection is close to last swapped center for this target
                            try:
                                if bool(control.get('DetectionLockToLastSwappedEnableToggle', True)):
                                    last_centers = getattr(self.video_processor, 'last_swapped_center', {}) or {}
                                    last_c = last_centers.get(target_face.face_id, None)
                                    bb = fface.get('bbox', None)
                                    if last_c is not None and isinstance(bb, np.ndarray) and bb.size >= 4:
                                        H = int(img.size()[1]); W = int(img.size()[2])
                                        diag = float((H*H + W*W) ** 0.5)
                                        near_pct = float(control.get('DetectionLockEaseRadiusPercentSlider', 0.4))
                                        near_R = max(1.5, (max(0.01, near_pct) / 100.0) * diag)
                                        cx = 0.5 * (float(bb[0]) + float(bb[2]))
                                        cy = 0.5 * (float(bb[1]) + float(bb[3]))
                                        d = ((cx - last_c[0])**2 + (cy - last_c[1])**2) ** 0.5
                                        if d <= near_R:
                                            ease_pts = float(control.get('DetectionLockEasePointsSlider', 4.0))
                                            eff_thr = max(0.0, eff_thr - ease_pts)
                            except Exception:
                                pass
                            if bool(control.get('AutoThresholdAssistEnableToggle', False)):
                                try:
                                    fid = target_face.face_id
                                    # Read current streak and offset
                                    streak = int(self.video_processor.recog_no_match_streak.get(fid, 0))
                                    offset = float(self.video_processor.recog_threshold_offset.get(fid, 0.0))
                                    # Read UI config with safe defaults
                                    max_drop = float(control.get('AutoThresholdAssistMaxDropSlider', 10))
                                    frames_per_step = max(1.0, float(control.get('AutoThresholdAssistFramesPerStepSlider', 15)))
                                    # Schedule: after each N consecutive no-match frames, lower threshold by 1 point, capped by max_drop
                                    planned_offset = -min(max_drop, (streak // frames_per_step) * 1.0)
                                    # Smoothly approach planned offset
                                    if planned_offset < offset:
                                        offset = planned_offset
                                    eff_thr = max(0.0, eff_thr + offset)
                                except Exception:
                                    pass
                            # AI Smart Tuning: transient easing
                            try:
                                if ai_tune_on and ai_thr_ease > 0.0:
                                    eff_thr = max(0.0, eff_thr - ai_thr_ease)
                            except Exception:
                                pass
                            # Track top similarity even if under threshold
                            if sim > top_best_sim:
                                top_best_sim = float(sim)
                                top_best_target_id = target_face.face_id
                            # Pixel-based force near last swapped center (independent of lock toggle)
                            force_near = False
                            try:
                                px = int(control.get('DetectionForceNearPxSlider', 0))
                                if px > 0:
                                    last_centers = getattr(self.video_processor, 'last_swapped_center', {}) or {}
                                    last_c = last_centers.get(target_face.face_id, None)
                                    bb = fface.get('bbox', None)
                                    if last_c is not None and isinstance(bb, np.ndarray) and bb.size >= 4:
                                        cx = 0.5 * (float(bb[0]) + float(bb[2]))
                                        cy = 0.5 * (float(bb[1]) + float(bb[3]))
                                        d = ((cx - last_c[0])**2 + (cy - last_c[1])**2) ** 0.5
                                        if d <= float(px):
                                            force_near = True
                            except Exception:
                                force_near = False
                            # Probe-under-threshold when sustained no-match streak and safe context
                            probe_ok = False
                            try:
                                fid = target_face.face_id
                                streak = int(self.video_processor.recog_no_match_streak.get(fid, 0))
                                if streak >= 8:
                                    probe_pts = min(12.0, 2.0 + 0.5 * streak)
                                    if sim >= max(0.0, eff_thr - probe_pts):
                                        # safer if only one detection or inside force radius
                                        if len(det_faces_data) == 1 or force_near:
                                            probe_ok = True
                            except Exception:
                                probe_ok = False
                            if force_near or (sim>=eff_thr) or probe_ok:
                                matched_count += 1
                                s_e = None
                                fface['kps_5'] = self.keypoints_adjustments(fface['kps_5'], parameters) #Make keypoints adjustments

                                # Estimate roll from 5-point landmarks and smooth it per target face
                                roll_deg = 0.0
                                try:
                                    dst_tmp = faceutil.get_arcface_template(image_size=512, mode='arcface128')
                                    dst_tmp = np.squeeze(dst_tmp)
                                    tform_tmp = trans.SimilarityTransform()
                                    tform_tmp.estimate(fface['kps_5'], dst_tmp)
                                    roll_deg = float(np.rad2deg(tform_tmp.rotation))
                                except Exception:
                                    pass

                                face_id = target_face.face_id
                                # Read UI controls for smoothing config
                                stab_enabled = bool(control.get('RotationStabilizationEnableToggle', True))
                                preset = str(control.get('RotationStabilizationPresetSelection', 'Medium'))
                                base_thr = float(control.get('RotationRollThresholdSlider', 3))
                                if not stab_enabled:
                                    engage_thr = 0.0
                                    release_thr = 0.0
                                else:
                                    # Hysteresis around threshold: engage at threshold, release slightly below
                                    engage_thr = max(0.0, base_thr)
                                    release_thr = max(0.0, base_thr - 1.0)

                                # Preset mapping for OneEuro
                                if preset == 'Low':
                                    min_cut = 1.0; beta = 0.005
                                elif preset == 'High':
                                    min_cut = 1.6; beta = 0.02
                                else:
                                    min_cut = 1.2; beta = 0.01

                                with self.video_processor.rotation_lock:
                                    filt = self.video_processor.rotation_filters.get(face_id)
                                    if filt is None:
                                        filt = faceutil.OneEuroFilter(freq=30.0, min_cutoff=min_cut, beta=beta, d_cutoff=1.0)
                                        self.video_processor.rotation_filters[face_id] = filt
                                    else:
                                        # Update params if preset changed
                                        filt.min_cutoff = float(min_cut)
                                        filt.beta = float(beta)
                                    engaged = self.video_processor.rotation_hysteresis.get(face_id, False)

                                # Use actual FPS for OneEuro when available (fallback to 30Hz)
                                try:
                                    freq = float(self.video_processor.fps) if getattr(self.video_processor, 'fps', 0) else 30.0
                                    if hasattr(filt, 'freq'):
                                        filt.freq = max(1.0, freq)
                                except Exception:
                                    pass
                                smoothed_roll = filt.filter(roll_deg)
                                if engaged:
                                    if abs(smoothed_roll) >= engage_thr:
                                        engaged = False
                                else:
                                    if abs(smoothed_roll) <= release_thr:
                                        engaged = True

                                with self.video_processor.rotation_lock:
                                    self.video_processor.rotation_hysteresis[face_id] = engaged

                                roll_override_deg = 0.0 if (stab_enabled and engaged) else float(smoothed_roll)
                                arcface_model = self.models_processor.get_arcface_model(parameters['SwapModelSelection'])
                                dfm_model=parameters['DFMModelSelection']
                                if self.main_window.swapfacesButton.isChecked():
                                    if parameters['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                        s_e = target_face.assigned_input_embedding.get(arcface_model, None)
                                    if s_e is not None and np.isnan(s_e).any():
                                        s_e = None
                                else:
                                    dfm_model = None
                                    s_e = None

                                # swap_core function is executed even if 'Swap Faces' button is disabled,
                                # because it also returns the original face and face mask 
                                attempted_count += 1
                                img, fface['original_face'], fface['swap_mask'], fface['tform'] = self.swap_core(
                                    img,
                                    fface['kps_5'],
                                    kps_all=fface.get('kps_all', None),
                                    s_e=s_e,
                                    t_e=target_face.get_embedding(arcface_model),
                                    parameters=parameters,
                                    control=control,
                                    dfm_model=dfm_model,
                                    roll_override_deg=roll_override_deg,
                                    face_id=face_id,
                                    ai_params={'on': ai_tune_on, 'feather_pct': ai_feather_pct, 'edge_strength': ai_edge_strength}
                                )
                                applied_count += 1
                                # Successful swap: clear relaxation early
                                try:
                                    self.video_processor.relax_lock_frames = 0
                                except Exception:
                                    pass
                                # Update last swapped center for this target to lock future detections
                                try:
                                    bb = fface.get('bbox', None)
                                    if isinstance(bb, np.ndarray) and bb.size >= 4:
                                        cx = 0.5 * (float(bb[0]) + float(bb[2]))
                                        cy = 0.5 * (float(bb[1]) + float(bb[3]))
                                        if not hasattr(self.video_processor, 'last_swapped_center'):
                                            self.video_processor.last_swapped_center = {}
                                        self.video_processor.last_swapped_center[face_id] = (cx, cy)
                                except Exception:
                                    pass
                                        # cv2.imwrite('temp_swap_face.png', swapped_face.permute(1,2,0).cpu().numpy())
                                if self.main_window.editFacesButton.isChecked():
                                    img = self.swap_edit_face_core(img, fface['kps_all'], parameters, control)
                            else:
                                # Update streaks for auto-threshold assist when enabled
                                if bool(control.get('AutoThresholdAssistEnableToggle', False)):
                                    try:
                                        fid = target_face.face_id
                                        s = int(self.video_processor.recog_no_match_streak.get(fid, 0)) + 1
                                        self.video_processor.recog_no_match_streak[fid] = s
                                    except Exception:
                                        pass
                        # Reset streaks on success (any applied swap)
                        try:
                            if applied_count > 0 and bool(control.get('AutoThresholdAssistEnableToggle', False)):
                                fid = target_face.face_id
                                # If there was a previous offset due to streak, log recovery and reset
                                try:
                                    from app.helpers.logger import log_threshold_tune
                                    base_thr = float(parameters['SimilarityThresholdSlider'])
                                    offset = float(self.video_processor.recog_threshold_offset.get(fid, 0.0))
                                    if offset != 0.0:
                                        log_threshold_tune(self.frame_number, fid, base_thr, base_thr + offset, offset, 0, 'reset_on_success')
                                except Exception:
                                    pass
                                self.video_processor.recog_no_match_streak[fid] = 0
                                self.video_processor.recog_threshold_offset[fid] = 0.0
                        except Exception:
                            pass

        # If no target face matched in this frame, attempt an aggressive re-detect + swap pass
        try:
            if (
                matched_count == 0
                and bool(control.get('DetectionAutoRetryEnableToggle', True))
                and self.main_window.swapfacesButton.isChecked()
            ):
                # Build more permissive detection settings
                retry_angles = [0, -10, 10, -20, 20, -35, 35, -50, 50] if control.get('AutoRotationToggle', True) else [0]
                base_det_score = float(control.get('DetectorScoreSlider', 50)) / 100.0
                retry_score = max(0.15, base_det_score - 0.15)
                retry_input = (640, 640)

                b2, k52, k2 = self.models_processor.run_detect(
                    img,
                    control['DetectorModelSelection'],
                    max_num=control['MaxFacesToDetectSlider'],
                    score=retry_score,
                    input_size=retry_input,
                    use_landmark_detection=use_landmark_detection,
                    landmark_detect_mode=landmark_detect_mode,
                    landmark_score=control["LandmarkDetectScoreSlider"]/100.0,
                    from_points=from_points,
                    rotation_angles=retry_angles
                )
                if isinstance(k52, np.ndarray) and k52.shape[0] > 0 and isinstance(b2, np.ndarray) and b2.shape[0] > 0:
                    det_faces_data2 = []
                    try:
                        for i in range(k52.shape[0]):
                            fk5 = np.asarray(k52[i], dtype=np.float32)
                            face_emb, _ = self.models_processor.run_recognize_direct(img, fk5, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                            face_kps_all = k2[i] if isinstance(k2, np.ndarray) and k2.ndim == 3 and k2.shape[0] > i else k52[i]
                            det_faces_data2.append({'kps_5': k52[i], 'kps_all': face_kps_all, 'embedding': face_emb, 'bbox': b2[i]})
                    except Exception:
                        det_faces_data2 = []

                    # Try to match and swap using same logic as primary pass
                    for fface in det_faces_data2:
                        for _, target_face in self.main_window.target_faces.items():
                            # Lock-to-last-swapped gating on retry pass as well
                            # Do not hard-gate by lock in retry pass either; still compute for easing/force and selection.
                            try:
                                _ = bool(control.get('DetectionLockToLastSwappedEnableToggle', True))
                                _ = int(getattr(self.video_processor, 'relax_lock_frames', 0))
                                _ = getattr(self.video_processor, 'last_swapped_center', {})
                            except Exception:
                                pass
                            # Safeguard parameters
                            if target_face.face_id not in self.parameters:
                                try:
                                    self.parameters[target_face.face_id] = self.main_window.default_parameters.copy()
                                except Exception:
                                    self.parameters[target_face.face_id] = dict(self.main_window.default_parameters)
                            parameters = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters)

                            sim = self.models_processor.findCosineDistance(fface['embedding'], target_face.get_embedding(control['RecognitionModelSelection']))
                            eff_thr = float(parameters['SimilarityThresholdSlider'])
                            if bool(control.get('AutoThresholdAssistEnableToggle', False)):
                                try:
                                    fid = target_face.face_id
                                    streak = int(self.video_processor.recog_no_match_streak.get(fid, 0))
                                    offset = float(self.video_processor.recog_threshold_offset.get(fid, 0.0))
                                    max_drop = float(control.get('AutoThresholdAssistMaxDropSlider', 10))
                                    frames_per_step = max(1.0, float(control.get('AutoThresholdAssistFramesPerStepSlider', 15)))
                                    planned_offset = -min(max_drop, (streak // frames_per_step) * 1.0)
                                    if planned_offset < offset:
                                        offset = planned_offset
                                    eff_thr = max(0.0, eff_thr + offset)
                                except Exception:
                                    pass
                            try:
                                if ai_tune_on and ai_thr_ease > 0.0:
                                    eff_thr = max(0.0, eff_thr - ai_thr_ease)
                            except Exception:
                                pass

                            if sim > top_best_sim:
                                top_best_sim = float(sim)
                                top_best_target_id = target_face.face_id

                            # Apply same pixel-based force condition on retry path
                            force_near2 = False
                            try:
                                if bool(control.get('DetectionLockToLastSwappedEnableToggle', True)):
                                    px = int(control.get('DetectionForceNearPxSlider', 0))
                                    if px > 0:
                                        last_centers = getattr(self.video_processor, 'last_swapped_center', {}) or {}
                                        last_c = last_centers.get(target_face.face_id, None)
                                        bb = fface.get('bbox', None)
                                        if last_c is not None and isinstance(bb, np.ndarray) and bb.size >= 4:
                                            cx = 0.5 * (float(bb[0]) + float(bb[2]))
                                            cy = 0.5 * (float(bb[1]) + float(bb[3]))
                                            d = ((cx - last_c[0])**2 + (cy - last_c[1])**2) ** 0.5
                                            if d <= float(px):
                                                force_near2 = True
                            except Exception:
                                force_near2 = False
                            # Probe-under-threshold on retry as well
                            probe_ok2 = False
                            try:
                                fid = target_face.face_id
                                streak = int(self.video_processor.recog_no_match_streak.get(fid, 0))
                                if streak >= 8:
                                    probe_pts = min(12.0, 2.0 + 0.5 * streak)
                                    if sim >= max(0.0, eff_thr - probe_pts):
                                        if len(det_faces_data2) == 1 or force_near2:
                                            probe_ok2 = True
                            except Exception:
                                probe_ok2 = False
                            if force_near2 or (sim >= eff_thr) or probe_ok2:
                                matched_count += 1
                                # Prepare smoothing and swap additional params (reuse defaults)
                                try:
                                    # Estimate roll
                                    dst_tmp = faceutil.get_arcface_template(image_size=512, mode='arcface128')
                                    dst_tmp = np.squeeze(dst_tmp)
                                    tform_tmp = trans.SimilarityTransform()
                                    tform_tmp.estimate(fface['kps_5'], dst_tmp)
                                    roll_deg = float(np.rad2deg(tform_tmp.rotation))
                                except Exception:
                                    roll_deg = 0.0

                                try:
                                    stab_enabled = bool(control.get('RotationStabilizationEnableToggle', True))
                                    preset = str(control.get('RotationStabilizationPresetSelection', 'Medium'))
                                    base_thr_roll = float(control.get('RotationRollThresholdSlider', 3))
                                    engage_thr = max(0.0, base_thr_roll) if stab_enabled else 0.0
                                    release_thr = max(0.0, base_thr_roll - 1.0) if stab_enabled else 0.0
                                    if preset == 'Low':
                                        min_cut = 1.0; beta = 0.005
                                    elif preset == 'High':
                                        min_cut = 1.6; beta = 0.02
                                    else:
                                        min_cut = 1.2; beta = 0.01
                                    face_id = target_face.face_id
                                    with self.video_processor.rotation_lock:
                                        filt = self.video_processor.rotation_filters.get(face_id)
                                        if filt is None:
                                            filt = faceutil.OneEuroFilter(freq=30.0, min_cutoff=min_cut, beta=beta, d_cutoff=1.0)
                                            self.video_processor.rotation_filters[face_id] = filt
                                        else:
                                            filt.min_cutoff = float(min_cut)
                                            filt.beta = float(beta)
                                        engaged = self.video_processor.rotation_hysteresis.get(face_id, False)
                                    try:
                                        freq = float(self.video_processor.fps) if getattr(self.video_processor, 'fps', 0) else 30.0
                                        if hasattr(filt, 'freq'):
                                            filt.freq = max(1.0, freq)
                                    except Exception:
                                        pass
                                    smoothed_roll = filt.filter(roll_deg)
                                    if engaged:
                                        if abs(smoothed_roll) >= engage_thr:
                                            engaged = False
                                    else:
                                        if abs(smoothed_roll) <= release_thr:
                                            engaged = True
                                    with self.video_processor.rotation_lock:
                                        self.video_processor.rotation_hysteresis[face_id] = engaged
                                    roll_override_deg = 0.0 if (stab_enabled and engaged) else float(smoothed_roll)
                                except Exception:
                                    roll_override_deg = 0.0

                                # Execute swap
                                attempted_count += 1
                                arcface_model = self.models_processor.get_arcface_model(parameters['SwapModelSelection'])
                                dfm_model = parameters['DFMModelSelection'] if self.main_window.swapfacesButton.isChecked() else None
                                s_e = None
                                if dfm_model is None and parameters['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                    s_e = target_face.assigned_input_embedding.get(arcface_model, None)
                                    if s_e is not None and np.isnan(s_e).any():
                                        s_e = None
                                img, fface['original_face'], fface['swap_mask'], fface['tform'] = self.swap_core(
                                    img,
                                    fface['kps_5'],
                                    kps_all=fface.get('kps_all', None),
                                    s_e=s_e,
                                    t_e=target_face.get_embedding(arcface_model),
                                    parameters=parameters,
                                    control=control,
                                    dfm_model=dfm_model,
                                    roll_override_deg=roll_override_deg,
                                    face_id=target_face.face_id,
                                    ai_params={'on': ai_tune_on, 'feather_pct': ai_feather_pct, 'edge_strength': ai_edge_strength}
                                )
                                applied_count += 1
                                # Successful swap: clear relaxation early
                                try:
                                    self.video_processor.relax_lock_frames = 0
                                except Exception:
                                    pass
                                # Update last swapped center to maintain lock
                                try:
                                    bb = fface.get('bbox', None)
                                    if isinstance(bb, np.ndarray) and bb.size >= 4:
                                        cx = 0.5 * (float(bb[0]) + float(bb[2]))
                                        cy = 0.5 * (float(bb[1]) + float(bb[3]))
                                        if not hasattr(self.video_processor, 'last_swapped_center'):
                                            self.video_processor.last_swapped_center = {}
                                        self.video_processor.last_swapped_center[target_face.face_id] = (cx, cy)
                                except Exception:
                                    pass
                                # On first success, update caches and break out to avoid extra work
                                try:
                                    self.video_processor.last_detections = {
                                        'bboxes': b2,
                                        'kpss_5': k52.copy(),
                                        'kpss': k2.copy() if isinstance(k2, np.ndarray) else k2,
                                        'frame_number': self.frame_number,
                                    }
                                    self.video_processor.last_input_frame_rgb = curr_frame_rgb
                                    self.video_processor.last_input_frame_number = self.frame_number
                                except Exception:
                                    pass
                                det_faces_data = det_faces_data2  # for overlays/compare
                                break
                        if applied_count > 0:
                            break
        except Exception:
            pass

        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=-control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        if control['ShowAllDetectedFacesBBoxToggle']:
            img = self.draw_bounding_boxes_on_detected_faces(img, det_faces_data, control)

        if control["ShowLandmarksEnableToggle"] and det_faces_data:
            img = img.permute(1,2,0)
            img = self.paint_face_landmarks(img, det_faces_data, control)
            img = img.permute(2,0,1)

        # Skip overlays/enhancer when no swap applied and performance fast-path is enabled
        if compare_mode:
            img = self.get_compare_faces_image(img, det_faces_data, control)

        if control['FrameEnhancerEnableToggle'] and not compare_mode:
            img = self.enhance_core(img, control=control)

        # Optional on-screen recognition debug overlay
        try:
            if bool(control.get('RecognitionDebugOverlayEnableToggle', False)):
                # Build overlay text
                lines = []
                # Include top similarity and base threshold of first target for context
                base_thr = None
                try:
                    if len(self.main_window.target_faces) > 0:
                        first_target = next(iter(self.main_window.target_faces.values()))
                        # Ensure parameters exist for this face id
                        if first_target.face_id not in self.parameters:
                            try:
                                self.parameters[first_target.face_id] = self.main_window.default_parameters.copy()
                            except Exception:
                                self.parameters[first_target.face_id] = dict(self.main_window.default_parameters)
                        base_thr_val = self.parameters[first_target.face_id].get('SimilarityThresholdSlider', self.main_window.default_parameters.get('SimilarityThresholdSlider', 0))
                        base_thr = float(base_thr_val)
                except Exception:
                    base_thr = None
                # If assist is on and we have a top target, compute eff_thr preview and show streak/offset
                eff_thr_preview = base_thr
                assist_on = bool(control.get('AutoThresholdAssistEnableToggle', False))
                if assist_on and top_best_target_id is not None and base_thr is not None:
                    try:
                        s = int(self.video_processor.recog_no_match_streak.get(top_best_target_id, 0))
                        off = float(self.video_processor.recog_threshold_offset.get(top_best_target_id, 0.0))
                        max_drop = float(control.get('AutoThresholdAssistMaxDropSlider', 10))
                        frames_per_step = max(1.0, float(control.get('AutoThresholdAssistFramesPerStepSlider', 15)))
                        planned = -min(max_drop, (s // frames_per_step) * 1.0)
                        # The effective threshold the worker will use next iteration is base+min(off, planned)
                        eff_thr_preview = max(0.0, base_thr + min(off, planned))
                        lines.append(f"Recog: sim={top_best_sim:.3f} base={base_thr:.1f} eff={eff_thr_preview:.1f} model={recog_model_name}")
                        lines.append(f"Assist: streak={s} offset={off:.1f} planned={planned:.1f}")
                    except Exception:
                        lines.append(f"Recog: sim={top_best_sim:.3f} model={recog_model_name}")
                else:
                    if base_thr is not None:
                        lines.append(f"Recog: sim={top_best_sim:.3f} thr={base_thr:.1f} model={recog_model_name}")
                    else:
                        lines.append(f"Recog: sim={top_best_sim:.3f} model={recog_model_name}")

                # Render overlay using cv2.putText for simplicity on a CPU tensor
                img_np = img.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                # Draw a semi-transparent background box
                try:
                    h, w, _ = img_np.shape
                    pad = 8
                    line_h = 20
                    box_h = pad * 2 + line_h * len(lines)
                    box_w = max(240, int(min(w * 0.6, 10 + max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines))))
                    overlay = img_np.copy()
                    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.4, img_np, 0.6, 0, img_np)
                except Exception:
                    pass
                y = pad + 16
                for t in lines:
                    try:
                        cv2.putText(img_np, t, (pad + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                    except Exception:
                        pass
                    y += 18
                # Optionally append AI Agent overlay line
                try:
                    if bool(control.get('AIAgentEnableToggle', False)) and bool(control.get('AIAgentDebugOverlayEnableToggle', False)):
                        agent = getattr(self.video_processor, 'ai_agent_state', {}) or {}
                        prof = agent.get('active_profile', None)
                        sm = getattr(self.video_processor, 'scene_metrics', {}) or {}
                        text = f"Agent: {prof or 'n/a'} | motion={sm.get('motion',0):.2f} light={sm.get('light',0):.2f} diff={sm.get('difficulty',0):.2f}"
                        cv2.putText(img_np, text, (pad + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
                        y += 18
                except Exception:
                    pass
                img = torch.from_numpy(img_np).to(self.models_processor.device).permute(2,0,1)
        except Exception:
            pass

        # Detailed logging for misses to match visual observations and update threshold offsets
        try:
            if detected_count > 0 and applied_count == 0:
                # Case 1: no face matched threshold
                if matched_count == 0:
                    # Log top similarity and the current threshold (note: thresholds per target may differ; we log the default of first target if available)
                    try:
                        first_params = None
                        if len(self.main_window.target_faces) > 0:
                            first_params = next(iter(self.main_window.target_faces.values()))
                        thr = None
                        if first_params is not None:
                            if first_params.face_id not in self.parameters:
                                try:
                                    self.parameters[first_params.face_id] = self.main_window.default_parameters.copy()
                                except Exception:
                                    self.parameters[first_params.face_id] = dict(self.main_window.default_parameters)
                            thr_val = self.parameters[first_params.face_id].get('SimilarityThresholdSlider', self.main_window.default_parameters.get('SimilarityThresholdSlider', 0))
                            thr = float(thr_val)
                    except Exception:
                        thr = None
                    extras = {
                        'detected': detected_count,
                        'matched': matched_count,
                        'attempted': attempted_count,
                        'applied': applied_count,
                        'top_sim': round(top_best_sim, 4) if top_best_sim >= 0 else None,
                        'top_target': top_best_target_id,
                        'thr': thr,
                        'recog_model': recog_model_name,
                        'reason': 'no_match_above_threshold'
                    }
                    log_swap_gap(self.frame_number, extras)
                    # Update per-target offset state when assist is enabled
                    if bool(control.get('AutoThresholdAssistEnableToggle', False)) and top_best_target_id is not None:
                        try:
                            fid = top_best_target_id
                            s = int(self.video_processor.recog_no_match_streak.get(fid, 0))
                            if fid in self.parameters:
                                base_thr_val2 = self.parameters[fid].get('SimilarityThresholdSlider', thr)
                                base_thr = float(base_thr_val2) if base_thr_val2 is not None else thr
                            else:
                                base_thr = thr
                            max_drop = float(control.get('AutoThresholdAssistMaxDropSlider', 10))
                            frames_per_step = max(1.0, float(control.get('AutoThresholdAssistFramesPerStepSlider', 15)))
                            planned_offset = -min(max_drop, (s // frames_per_step) * 1.0)
                            prev = float(self.video_processor.recog_threshold_offset.get(fid, 0.0))
                            if planned_offset != prev:
                                self.video_processor.recog_threshold_offset[fid] = planned_offset
                                from app.helpers.logger import log_threshold_tune
                                log_threshold_tune(self.frame_number, fid, base_thr, base_thr + planned_offset, planned_offset, s, 'increase_offset')
                        except Exception:
                            pass
                # Case 2: there were matches but still nothing applied (pipeline block)
                elif matched_count > 0:
                    extras = {
                        'detected': detected_count,
                        'matched': matched_count,
                        'attempted': attempted_count,
                        'applied': applied_count,
                        'reason': 'match_but_not_applied'
                    }
                    log_swap_gap(self.frame_number, extras)
        except Exception:
            pass

        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        # RGB to BGR
        return img[..., ::-1]

    def _track_kps_lk(self, prev_rgb: np.ndarray, curr_rgb: np.ndarray, last_kpss_5: np.ndarray, last_bboxes: np.ndarray):
        """Track 5-point landmarks per face using pyramidal LK between prev and current RGB frames.
        Returns dict with 'kpss_5' and 'bboxes' if successful, else None.
        """
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        new_kps_list = []
        new_bbox_list = []
        for i in range(last_kpss_5.shape[0]):
            pts = last_kpss_5[i].astype(np.float32).reshape(-1, 1, 2)
            nxt, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None,
                                                    winSize=(21, 21), maxLevel=3,
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if nxt is None or st is None:
                continue
            good = st.reshape(-1) == 1
            if good.sum() < 3:
                continue
            tracked_pts = nxt.reshape(-1, 2)
            if good.sum() < 5:
                tracked_pts[~good] = last_kpss_5[i][~good]
            new_kps_list.append(tracked_pts)
            # derive bbox from tracked points, expand by 1.6x relative to their span
            x_min, y_min = tracked_pts[:,0].min(), tracked_pts[:,1].min()
            x_max, y_max = tracked_pts[:,0].max(), tracked_pts[:,1].max()
            cx, cy = (x_min+x_max)/2.0, (y_min+y_max)/2.0
            w, h = (x_max-x_min), (y_max-y_min)
            scale = 1.6
            bw, bh = w*scale, h*scale
            x0, y0 = int(max(0, cx - bw/2)), int(max(0, cy - bh/2))
            x1, y1 = int(min(curr_rgb.shape[1]-1, cx + bw/2)), int(min(curr_rgb.shape[0]-1, cy + bh/2))
            new_bbox_list.append([x0, y0, x1, y1])

        if not new_kps_list:
            return None

        return {
            'kpss_5': np.stack(new_kps_list, axis=0).astype(np.float32),
            'bboxes': np.array(new_bbox_list, dtype=np.float32)
        }

    def _transform_all_landmarks(self, prev_kpss_all: np.ndarray, prev_kpss_5: np.ndarray, new_kpss_5: np.ndarray) -> np.ndarray:
        """Apply a per-face similarity transform estimated from 5 points to dense landmarks."""
        out = []
        for i in range(new_kpss_5.shape[0]):
            tform = trans.SimilarityTransform()
            if i >= prev_kpss_5.shape[0] or i >= prev_kpss_all.shape[0]:
                continue
            try:
                tform.estimate(prev_kpss_5[i], new_kpss_5[i])
                pts = prev_kpss_all[i]
                ones = np.ones((pts.shape[0], 1), dtype=np.float32)
                hom = np.hstack([pts, ones])
                new_pts = (hom @ tform.params[:2].T)
            except Exception:
                new_pts = prev_kpss_all[i]
            out.append(new_pts)
        if not out:
            return prev_kpss_all
        return np.stack(out, axis=0).astype(np.float32)
    
    def keypoints_adjustments(self, kps_5: np.ndarray, parameters: dict) -> np.ndarray:
        # Change the ref points
        if parameters['FaceAdjEnableToggle']:
            kps_5[:,0] += parameters['KpsXSlider']
            kps_5[:,1] += parameters['KpsYSlider']
            kps_5[:,0] -= 255
            kps_5[:,0] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,0] += 255
            kps_5[:,1] -= 255
            kps_5[:,1] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,1] += 255

        # Face Landmarks
        if parameters['LandmarksPositionAdjEnableToggle']:
            kps_5[0][0] += parameters['EyeLeftXAmountSlider']
            kps_5[0][1] += parameters['EyeLeftYAmountSlider']
            kps_5[1][0] += parameters['EyeRightXAmountSlider']
            kps_5[1][1] += parameters['EyeRightYAmountSlider']
            kps_5[2][0] += parameters['NoseXAmountSlider']
            kps_5[2][1] += parameters['NoseYAmountSlider']
            kps_5[3][0] += parameters['MouthLeftXAmountSlider']
            kps_5[3][1] += parameters['MouthLeftYAmountSlider']
            kps_5[4][0] += parameters['MouthRightXAmountSlider']
            kps_5[4][1] += parameters['MouthRightYAmountSlider']
        return kps_5
    
    def paint_face_landmarks(self, img: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        # if img_y <= 720:
        #     p = 1
        # else:
        #     p = 2
        p = 2 #Point thickness
        for i, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                if target_face.face_id not in self.parameters:
                    try:
                        self.parameters[target_face.face_id] = self.main_window.default_parameters.copy()
                    except Exception:
                        self.parameters[target_face.face_id] = dict(self.main_window.default_parameters)
                # Wrap in ParametersDict for default fallback behavior
                parameters = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters) #Use the parameters of the target face
                sim = self.models_processor.findCosineDistance(fface['embedding'], target_face.get_embedding(control['RecognitionModelSelection']))
                if sim>=parameters['SimilarityThresholdSlider']:
                    if parameters['LandmarksPositionAdjEnableToggle']:
                        kcolor = tuple((255, 0, 0))
                        keypoints = fface['kps_5']
                    else:
                        kcolor = tuple((0, 255, 255))
                        keypoints = fface['kps_all']

                    for kpoint in keypoints:
                        for i in range(-1, p):
                            for j in range(-1, p):
                                try:
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][0] = kcolor[0]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][1] = kcolor[1]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][2] = kcolor[2]

                                except ValueError:
                                    #print("Key-points value {} exceed the image size {}.".format(kpoint, (img_x, img_y)))
                                    continue
        return img
    
    def draw_bounding_boxes_on_detected_faces(self, img: torch.Tensor, det_faces_data: list, control: dict):
        for i, fface in enumerate(det_faces_data):
            color = [0, 255, 0]
            bbox = fface['bbox']
            x_min, y_min, x_max, y_max = map(int, bbox)
            # Ensure bounding box is within the image dimensions
            _, h, w = img.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
            # Dynamically compute thickness based on the image resolution
            max_dimension = max(img.shape[1], img.shape[2])  # Height and width of the image
            thickness = max(4, max_dimension // 400)  # Thickness is 1/200th of the largest dimension, minimum 1
            # Prepare the color tensor with the correct dimensions
            color_tensor = torch.tensor(color, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            # Draw the top edge
            img[:, y_min:y_min + thickness, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            # Draw the bottom edge
            img[:, y_max - thickness + 1:y_max + 1, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            # Draw the left edge
            img[:, y_min:y_max + 1, x_min:x_min + thickness] = color_tensor.expand(-1, y_max - y_min + 1, thickness)
            # Draw the right edge
            img[:, y_min:y_max + 1, x_max - thickness + 1:x_max + 1] = color_tensor.expand(-1, y_max - y_min + 1, thickness)   
        return img

    def get_compare_faces_image(self, img: torch.Tensor, det_faces_data: dict, control: dict) -> torch.Tensor:
        imgs_to_vstack = []  # Renamed for vertical stacking
        for _, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                if target_face.face_id not in self.parameters:
                    try:
                        self.parameters[target_face.face_id] = self.main_window.default_parameters.copy()
                    except Exception:
                        self.parameters[target_face.face_id] = dict(self.main_window.default_parameters)
                parameters = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters)  # Use the parameters of the target face
                sim = self.models_processor.findCosineDistance(
                    fface['embedding'], 
                    target_face.get_embedding(control['RecognitionModelSelection'])
                )
                if sim >= parameters['SimilarityThresholdSlider']:
                    # Use the same transform used for swapping (keeps orientation/centering identical)
                    if fface.get('tform') is not None:
                        tform = fface['tform']
                        modified_face = v2.functional.affine(
                            img,
                            tform.rotation * 57.2958,
                            (tform.translation[0], tform.translation[1]),
                            tform.scale,
                            0,
                            center=(0, 0),
                            interpolation=v2.InterpolationMode.BILINEAR,
                        )
                        modified_face = v2.functional.crop(modified_face, 0, 0, 512, 512)
                    else:
                        # Fallback: recompute from KPS if tform wasn't produced
                        modified_face = self.get_cropped_face_using_kps(img, fface['kps_5'], parameters)
                    # Apply frame enhancer
                    if control['FrameEnhancerEnableToggle']:
                        # Enhance the face and resize it to the original size for stacking
                        modified_face_enhance = self.enhance_core(modified_face, control=control)
                        modified_face_enhance = modified_face_enhance.float() / 255.0
                        # Resize source_tensor to match the size of target_tensor
                        modified_face = torch.functional.F.interpolate(
                            modified_face_enhance.unsqueeze(0),  # Add batch dimension
                            size=modified_face.shape[1:],  # Target size: [H, W]
                            mode='bilinear',  # Interpolation mode
                            align_corners=False  # Avoid alignment artifacts
                        ).squeeze(0)  # Remove batch dimension
                        
                        modified_face = (modified_face * 255).clamp(0, 255).to(dtype=torch.uint8)
                    imgs_to_cat = []
                    
                    # Append tensors to imgs_to_cat
                    if fface['original_face'] is not None:
                        imgs_to_cat.append(fface['original_face'].permute(2, 0, 1))
                    imgs_to_cat.append(modified_face)
                    if fface['swap_mask'] is not None:
                        fface['swap_mask'] = 255-fface['swap_mask']
                        imgs_to_cat.append(fface['swap_mask'].permute(2, 0, 1))
  
                    # Concatenate horizontally for comparison
                    img_compare = torch.cat(imgs_to_cat, dim=2)

                    # Add horizontally concatenated image to vertical stack list
                    imgs_to_vstack.append(img_compare)
    
        if imgs_to_vstack:
            # Find the maximum width
            max_width = max(img_to_stack.size(2) for img_to_stack in imgs_to_vstack)
            
            # Pad images to have the same width
            padded_imgs = [
                torch.nn.functional.pad(img_to_stack, (0, max_width - img_to_stack.size(2), 0, 0)) 
                for img_to_stack in imgs_to_vstack
            ]
            # Stack images vertically
            img_vstack = torch.cat(padded_imgs, dim=1)  # Use dim=1 for vertical stacking
            img = img_vstack
        return img
        
    def get_cropped_face_using_kps(self, img: torch.Tensor, kps_5: np.ndarray, parameters: dict) -> torch.Tensor:
        tform = self.get_face_similarity_tform(parameters['SwapModelSelection'], kps_5)
        # Grab 512 face from image and create 256 and 128 copys
        face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR )
        face_512 = v2.functional.crop(face_512, 0,0, 512, 512)# 3, 512, 512
        return face_512

    def get_face_similarity_tform(self, swapper_model: str, kps_5: np.ndarray, roll_override_deg: float = None, *, face_id: str | None = None, control: dict | None = None) -> trans.SimilarityTransform:
        tform = trans.SimilarityTransform()
        if swapper_model != 'GhostFace-v1' and swapper_model != 'GhostFace-v2' and swapper_model != 'GhostFace-v3' and swapper_model != 'CSCS':
            dst = faceutil.get_arcface_template(image_size=512, mode='arcface128')
            dst = np.squeeze(dst)
            # First estimate full similarity from src->dst
            tform.estimate(kps_5, dst)
            # If we override roll, rebuild a consistent transform: keep scale but recompute translation for the new rotation
            if roll_override_deg is not None:
                s = float(tform.scale)
                theta = float(np.deg2rad(roll_override_deg))
                c, si = np.cos(theta), np.sin(theta)
                A = np.array([[s*c, -s*si], [s*si, s*c]], dtype=np.float32)
                src_c = np.asarray(kps_5, dtype=np.float32).mean(axis=0)
                dst_c = np.asarray(dst, dtype=np.float32).mean(axis=0)
                # translation so that centroids map: t = dst_c - A @ src_c
                t = dst_c - (A @ src_c)
                tform = trans.SimilarityTransform(scale=s, rotation=theta, translation=t)
        elif swapper_model == "CSCS":
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else:
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            tform.params[0:2] = M

        # Optional temporal smoothing for translation/scale
        try:
            if control is None:
                control = {}
            stab_enabled = bool(control.get('RotationStabilizationEnableToggle', True))
            if stab_enabled and face_id is not None:
                preset = str(control.get('RotationStabilizationPresetSelection', 'Medium'))
                if preset == 'Low':
                    min_cut = 1.0; beta = 0.005
                elif preset == 'High':
                    min_cut = 1.6; beta = 0.02
                else:
                    min_cut = 1.2; beta = 0.01
                with self.video_processor.rotation_lock:
                    filt_set = self.video_processor.transform_filters.get(face_id)
                    # Use actual FPS when available
                    try:
                        freq = float(self.video_processor.fps) if getattr(self.video_processor, 'fps', 0) else 30.0
                        freq = max(1.0, freq)
                    except Exception:
                        freq = 30.0
                    if filt_set is None:
                        filt_set = {
                            'tx': faceutil.OneEuroFilter(freq=freq, min_cutoff=min_cut, beta=beta, d_cutoff=1.0),
                            'ty': faceutil.OneEuroFilter(freq=freq, min_cutoff=min_cut, beta=beta, d_cutoff=1.0),
                            's': faceutil.OneEuroFilter(freq=freq, min_cutoff=min_cut, beta=beta, d_cutoff=1.0),
                        }
                        self.video_processor.transform_filters[face_id] = filt_set
                    else:
                        for k in ('tx','ty','s'):
                            filt_set[k].min_cutoff = float(min_cut)
                            filt_set[k].beta = float(beta)
                            if hasattr(filt_set[k], 'freq'):
                                filt_set[k].freq = freq
                # Filter current values
                tx = float(tform.translation[0]); ty = float(tform.translation[1]); s = float(tform.scale)
                # Keep tx/ty filters for future use, but we will recenter via centroid to prevent off-center crops
                _tx_f = filt_set['tx'].filter(tx)
                _ty_f = filt_set['ty'].filter(ty)
                s_f = max(1e-5, filt_set['s'].filter(s))
                # Apply a small deadband to reduce micro-jitter
                try:
                    prev_tx = getattr(filt_set['tx'], 'last_raw', None)
                    prev_ty = getattr(filt_set['ty'], 'last_raw', None)
                    prev_s  = getattr(filt_set['s'],  'last_raw', None)
                    # We store last filtered outputs to compare
                    prev_tx_f = getattr(filt_set['tx'], 'last_filtered', None)
                    prev_ty_f = getattr(filt_set['ty'], 'last_filtered', None)
                    prev_s_f  = getattr(filt_set['s'],  'last_filtered', None)
                except Exception:
                    prev_tx = prev_ty = prev_s = prev_tx_f = prev_ty_f = prev_s_f = None
                # Deadband thresholds (in px and unitless scale)
                db_tx_ty = 0.25
                db_s = 0.001
                if prev_s_f is not None and abs(s_f - prev_s_f) < db_s:
                    s_f = prev_s_f

                # Adaptive snap on fast motion: if raw jump vs previous filtered is large, bypass smoothing this frame
                try:
                    snap_thr_px = 3.0  # pixels
                    snap_thr_s  = 0.01  # scale units
                    if prev_tx_f is not None and prev_ty_f is not None:
                        dx = float(tx - prev_tx_f)
                        dy = float(ty - prev_ty_f)
                        motion = (dx*dx + dy*dy) ** 0.5
                    else:
                        motion = 0.0
                    ds = abs(float(s - (prev_s_f if prev_s_f is not None else s)))
                    if motion > snap_thr_px or ds > snap_thr_s:
                        s_f = max(1e-5, s)
                except Exception:
                    pass
                # Update helpers if attributes exist (store last filtered outputs)
                try:
                    filt_set['tx'].last_filtered = _tx_f
                    filt_set['ty'].last_filtered = _ty_f
                    filt_set['s'].last_filtered  = s_f
                except Exception:
                    pass
                # Recenter using centroid mapping to keep face centered in the 512 crop
                try:
                    theta = float(tform.rotation)
                    c, si = np.cos(theta), np.sin(theta)
                    R = np.array([[c, -si], [si, c]], dtype=np.float32)
                    s0 = float(s)
                    A0 = s0 * R
                    # Source centroid in image coords
                    src_c = np.asarray(kps_5, dtype=np.float32).mean(axis=0)
                    # Current target centroid under original tform
                    dst_c0 = (A0 @ src_c) + np.array([tx, ty], dtype=np.float32)
                    # New translation that maps src_c to the same dst_c0 with smoothed scale
                    Af = float(s_f) * R
                    t_new = dst_c0 - (Af @ src_c)
                    tform = trans.SimilarityTransform(scale=s_f, rotation=theta, translation=(float(t_new[0]), float(t_new[1])))
                except Exception:
                    tform = trans.SimilarityTransform(scale=s_f, rotation=float(tform.rotation), translation=(tx, ty))
                # Mark face_id as active for cleanup bookkeeping
                try:
                    import time
                    self.video_processor._filter_last_seen[face_id] = time.perf_counter()
                except Exception:
                    pass
        except Exception:
            pass
        return tform
      
    def get_transformed_and_scaled_faces(self, tform, img):
        # Grab 512 face from image and create 256 and 128 copys
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=v2.InterpolationMode.BILINEAR )
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)# 3, 512, 512
        original_face_384 = t384(original_face_512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        return original_face_512, original_face_384, original_face_256, original_face_128
    
    def get_affined_face_dim_and_swapping_latents(self, original_faces: tuple, swapper_model, dfm_model, s_e, t_e, parameters,):
        original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        if swapper_model == 'Inswapper128':
            self.models_processor.load_inswapper_iss_emap('Inswapper128')
            latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 1
            if parameters['SwapperResSelection'] == '128':
                dim = 1
                input_face_affined = original_face_128
            elif parameters['SwapperResSelection'] == '256':
                dim = 2
                input_face_affined = original_face_256
            elif parameters['SwapperResSelection'] == '384':
                dim = 3
                input_face_affined = original_face_384
            elif parameters['SwapperResSelection'] == '512':
                dim = 4
                input_face_affined = original_face_512

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            self.models_processor.load_inswapper_iss_emap(swapper_model)
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(s_e, version)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(t_e, version)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'SimSwap512':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 4
            input_face_affined = original_face_512

        elif swapper_model == 'GhostFace-v1' or swapper_model == 'GhostFace-v2' or swapper_model == 'GhostFace-v3':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'CSCS':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle']:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 2
            input_face_affined = original_face_256

        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            dfm_model = self.models_processor.load_dfm_model(dfm_model)
            latent = []
            input_face_affined = original_face_512
            dim = 4
        return input_face_affined, dfm_model, dim, latent
    
    def get_swapped_and_prev_face(self, output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters, ):
        # original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        prev_face = input_face_affined.clone()
        if swapper_model == 'Inswapper128':
            with torch.no_grad():  # Disabilita il calcolo del gradiente se è solo per inferenza
                for _ in range(itex):
                    for j in range(dim):
                        for i in range(dim):
                            input_face_disc = input_face_affined[j::dim,i::dim]
                            input_face_disc = input_face_disc.permute(2, 0, 1)
                            input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()

                            swapper_output = torch.empty((1,3,128,128), dtype=torch.float32, device=self.models_processor.device).contiguous()
                            self.models_processor.run_inswapper(input_face_disc, latent, swapper_output)

                            swapper_output = torch.squeeze(swapper_output)
                            swapper_output = swapper_output.permute(1, 2, 0)

                            output[j::dim, i::dim] = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1] #Version Name
            with torch.no_grad():  # Disabilita il calcolo del gradiente se è solo per inferenza
                for _ in range(itex):
                    input_face_disc = input_face_affined.permute(2, 0, 1)
                    input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()

                    swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                    self.models_processor.run_iss_swapper(input_face_disc, latent, swapper_output, version)

                    swapper_output = torch.squeeze(swapper_output)
                    swapper_output = swapper_output.permute(1, 2, 0)

                    output = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255)
                    output = torch.clamp(output, 0, 255)

        elif swapper_model == 'SimSwap512':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_simswap512(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'GhostFace-v1' or swapper_model == 'GhostFace-v2' or swapper_model == 'GhostFace-v3':
            for k in range(itex):
                input_face_disc = torch.mul(input_face_affined, 255.0).permute(2, 0, 1)
                input_face_disc = torch.div(input_face_disc.float(), 127.5)
                input_face_disc = torch.sub(input_face_disc, 1)
                #input_face_disc = input_face_disc[[2, 1, 0], :, :] # Inverte i canali da BGR a RGB (assumendo che l'input sia BGR)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_ghostface(input_face_disc, latent, swapper_output, swapper_model)
                swapper_output = swapper_output[0]
                swapper_output = swapper_output.permute(1, 2, 0)
                swapper_output = torch.mul(swapper_output, 127.5)
                swapper_output = torch.add(swapper_output, 127.5)
                #swapper_output = swapper_output[:, :, [2, 1, 0]] # Inverte i canali da RGB a BGR (assumendo che l'input sia RGB)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                input_face_affined = torch.div(input_face_affined, 255)

                output = swapper_output.clone()
                output = torch.clamp(output, 0, 255)

        elif swapper_model == 'CSCS':
            for k in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1)
                input_face_disc = v2.functional.normalize(input_face_disc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
                input_face_disc = torch.unsqueeze(input_face_disc, 0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_cscs(input_face_disc, latent, swapper_output)
                swapper_output = torch.squeeze(swapper_output)
                swapper_output = torch.add(torch.mul(swapper_output, 0.5), 0.5)
                swapper_output = swapper_output.permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()

                output = swapper_output.clone()
                output = torch.mul(output, 255)
                output = torch.clamp(output, 0, 255)
        
        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            out_celeb, _, _ = dfm_model.convert(original_face_512, parameters['DFMAmpMorphSlider']/100, rct=parameters['DFMRCTColorToggle'])
            prev_face = input_face_affined.clone()
            input_face_affined = out_celeb.clone()
            output = out_celeb.clone()

        output = output.permute(2, 0, 1)
        swap = t512(output)
        return swap, prev_face
    
    def get_border_mask(self, parameters):
        # Create border mask
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        border_mask = torch.unsqueeze(border_mask,0)

        # if parameters['BorderState']:
        top = parameters['BorderTopSlider']
        left = parameters['BorderLeftSlider']
        right = 128 - parameters['BorderRightSlider']
        bottom = 128 - parameters['BorderBottomSlider']

        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0

        gauss = transforms.GaussianBlur(parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2)
        border_mask = gauss(border_mask)
        return border_mask
            
    def swap_core(self, img, kps_5, kps=False, s_e=None, t_e=None, parameters=None, control=None, dfm_model=False, *, roll_override_deg: float = None, face_id: str | None = None, ai_params: dict | None = None, kps_all: np.ndarray | None = None): # img = RGB
        s_e = s_e if isinstance(s_e, np.ndarray) else []
        t_e = t_e if isinstance(t_e, np.ndarray) else []
        parameters = parameters or {}
        control = control or {}
        # parameters = self.parameters.copy()
        # AI Smart Tuning (transient) defaults
        ai_tune_on_local = False
        ai_feather_pct_local = 0.0
        ai_edge_strength_local = 0.0
        try:
            if isinstance(ai_params, dict):
                ai_tune_on_local = bool(ai_params.get('on', False))
                ai_feather_pct_local = float(ai_params.get('feather_pct', 0.0))
                ai_edge_strength_local = float(ai_params.get('edge_strength', 0.0))
        except Exception:
            ai_tune_on_local = False
            ai_feather_pct_local = 0.0
            ai_edge_strength_local = 0.0
        swapper_model = parameters['SwapModelSelection']

        tform = self.get_face_similarity_tform(swapper_model, kps_5, roll_override_deg, face_id=face_id, control=control)

        # Grab 512 face from image and create 256 and 128 copys
        original_face_512, original_face_384, original_face_256, original_face_128 = self.get_transformed_and_scaled_faces(tform, img)
        original_faces = (original_face_512, original_face_384, original_face_256, original_face_128)
        dim=1
        if (s_e is not None and len(s_e) > 0) or (swapper_model == 'DeepFaceLive (DFM)' and dfm_model):

            input_face_affined, dfm_model, dim, latent = self.get_affined_face_dim_and_swapping_latents(original_faces, swapper_model, dfm_model, s_e, t_e, parameters)

            # Optional Scaling # change the transform matrix scaling from center
            if parameters['FaceAdjEnableToggle']:
                input_face_affined = v2.functional.affine(input_face_affined, 0, (0, 0), 1 + parameters['FaceScaleAmountSlider'] / 100, 0, center=(dim*128/2, dim*128/2), interpolation=v2.InterpolationMode.BILINEAR)

            itex = 1
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)

            # Create empty output image and preprocess it for swapping
            output_size = int(128 * dim)
            output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
            input_face_affined = input_face_affined.permute(1, 2, 0)
            input_face_affined = torch.div(input_face_affined, 255.0)

            swap, prev_face = self.get_swapped_and_prev_face(output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters)
        
        else:
            swap = original_face_512
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)
                prev_face = torch.div(swap, 255.)
                prev_face = prev_face.permute(1, 2, 0)

        if parameters['StrengthEnableToggle']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthAmountSlider'], 100)*0.01
                if alpha==0:
                    alpha=1

                # Blend the images
                prev_face = torch.mul(prev_face, 255)
                prev_face = torch.clamp(prev_face, 0, 255)
                prev_face = prev_face.permute(2, 0, 1)
                prev_face = t512(prev_face)
                swap = torch.mul(swap, alpha)
                prev_face = torch.mul(prev_face, 1-alpha)
                swap = torch.add(swap, prev_face)

        border_mask = self.get_border_mask(parameters)

        # Create image mask
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device)
        swap_mask = torch.unsqueeze(swap_mask,0)
        
        # Expression Restorer
        if parameters['FaceExpressionEnableToggle']:
            swap = self.apply_face_expression_restorer(original_face_512, swap, parameters)

        # Restorer
        if parameters["FaceRestorerEnableToggle"]:
            swap = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], control['DetectorScoreSlider'])

        # Restorer2
        if parameters["FaceRestorerEnable2Toggle"]:
            swap = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], control['DetectorScoreSlider'])

        # Occluder
        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.apply_occlusion(original_face_256, parameters["OccluderSizeSlider"])
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)

        if parameters["DFLXSegEnableToggle"]:
            img_mask = self.models_processor.apply_dfl_xseg(original_face_256, -parameters["DFLXSegSizeSlider"])
            img_mask = t128(img_mask)
            swap_mask = torch.mul(swap_mask, 1 - img_mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)

        if parameters["FaceParserEnableToggle"]:
            #cv2.imwrite('swap.png', cv2.cvtColor(swap.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
            mask = self.models_processor.apply_face_parser(swap, parameters)
            mask = t128(mask)
            swap_mask = torch.mul(swap_mask, mask)

        # CLIPs
        if parameters["ClipEnableToggle"]:
            mask = self.models_processor.run_CLIPs(original_face_512, parameters["ClipText"], parameters["ClipAmountSlider"])
            mask = t128(mask)
            swap_mask *= mask

        if parameters['RestoreMouthEnableToggle'] or parameters['RestoreEyesEnableToggle']:
            M = tform.params[0:2]
            ones_column = np.ones((kps_5.shape[0], 1), dtype=np.float32)
            homogeneous_kps = np.hstack([kps_5, ones_column])
            dst_kps_5 = np.dot(homogeneous_kps, M.T)

            img_swap_mask = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
            img_orig_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()

            if parameters['RestoreMouthEnableToggle']:
                img_swap_mask = self.models_processor.restore_mouth(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreMouthBlendAmountSlider']/100, parameters['RestoreMouthFeatherBlendSlider'], parameters['RestoreMouthSizeFactorSlider']/100, parameters['RestoreXMouthRadiusFactorDecimalSlider'], parameters['RestoreYMouthRadiusFactorDecimalSlider'], parameters['RestoreXMouthOffsetSlider'], parameters['RestoreYMouthOffsetSlider'])
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            if parameters['RestoreEyesEnableToggle']:
                img_swap_mask = self.models_processor.restore_eyes(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreEyesBlendAmountSlider']/100, parameters['RestoreEyesFeatherBlendSlider'], parameters['RestoreEyesSizeFactorDecimalSlider'],  parameters['RestoreXEyesRadiusFactorDecimalSlider'], parameters['RestoreYEyesRadiusFactorDecimalSlider'], parameters['RestoreXEyesOffsetSlider'], parameters['RestoreYEyesOffsetSlider'], parameters['RestoreEyesSpacingOffsetSlider'])
                img_swap_mask = torch.clamp(img_swap_mask, 0, 1)

            gauss = transforms.GaussianBlur(parameters['RestoreEyesMouthBlurSlider']*2+1, (parameters['RestoreEyesMouthBlurSlider']+1)*0.2)
            img_swap_mask = gauss(img_swap_mask)

            img_swap_mask = t128(img_swap_mask)
            swap_mask = torch.mul(swap_mask, img_swap_mask)

        # Face Diffing
        if parameters["DifferencingEnableToggle"]:
            mask = self.models_processor.apply_fake_diff(swap, original_face_512, parameters["DifferencingAmountSlider"])
            gauss = transforms.GaussianBlur(parameters['DifferencingBlendAmountSlider']*2+1, (parameters['DifferencingBlendAmountSlider']+1)*0.2)
            mask = gauss(mask.type(torch.float32))
            swap = swap * mask + original_face_512*(1-mask)

        if parameters["AutoColorEnableToggle"]:
            # Histogram color matching original face on swapped face
            if parameters['AutoColorTransferTypeSelection'] == 'Test':
                swap = faceutil.histogram_matching(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'Test_Mask':
                swap = faceutil.histogram_matching_withmask(original_face_512, swap, t512(swap_mask), parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Test':
                swap = faceutil.histogram_matching_DFL_test(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])

            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, t512(swap_mask), parameters["AutoColorBlendAmountSlider"])

        # Apply color corrections
        if parameters['ColorEnableToggle']:
            swap = torch.unsqueeze(swap,0).contiguous()
            swap = v2.functional.adjust_gamma(swap, parameters['ColorGammaDecimalSlider'], 1.0)
            swap = torch.squeeze(swap)
            swap = swap.permute(1, 2, 0).type(torch.float32)

            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=self.models_processor.device)
            swap += del_color
            swap = torch.clamp(swap, min=0., max=255.)
            swap = swap.permute(2, 0, 1).type(torch.uint8)

            swap = v2.functional.adjust_brightness(swap, parameters['ColorBrightnessDecimalSlider'])
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastDecimalSlider'])
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationDecimalSlider'])
            swap = v2.functional.adjust_sharpness(swap, parameters['ColorSharpnessDecimalSlider'])
            swap = v2.functional.adjust_hue(swap, parameters['ColorHueDecimalSlider'])

            if parameters['ColorNoiseDecimalSlider'] > 0:
                swap = swap.permute(1, 2, 0).type(torch.float32)
                swap = swap + parameters['ColorNoiseDecimalSlider']*torch.randn(512, 512, 3, device=self.models_processor.device)
                swap = torch.clamp(swap, 0, 255)
                swap = swap.permute(2, 0, 1)

        if parameters['JPEGCompressionEnableToggle']:
            try:
                swap = faceutil.jpegBlur(swap, parameters["JPEGCompressionAmountSlider"])
            except:
                pass
        if parameters['FinalBlendAdjEnableToggle'] and parameters['FinalBlendAdjEnableToggle'] > 0:
            final_blur_strength = parameters['FinalBlendAmountSlider']  # Ein Parameter steuert beides
            # Bestimme kernel_size und sigma basierend auf dem Parameter
            kernel_size = 2 * final_blur_strength + 1  # Ungerade Zahl, z.B. 3, 5, 7, ...
            sigma = final_blur_strength * 0.1  # Sigma proportional zur Stärke
            # Gaussian Blur anwenden
            gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
            swap = gaussian_blur(swap)

        # Auto Pitch Compensation (directional mask attenuation top/bottom)
        try:
            if parameters.get('AutoPitchCompEnableToggle', False):
                strength_ui = float(parameters.get('AutoPitchCompStrengthSlider', 0)) / 100.0
                if strength_ui > 0.0 and isinstance(kps_5, np.ndarray) and kps_5.shape == (5, 2):
                    # Remove roll from kps_5 to get a more stable vertical reference
                    theta = float(tform.rotation) if hasattr(tform, 'rotation') else 0.0
                    c, s = np.cos(-theta), np.sin(-theta)
                    R = np.array([[c, -s], [s, c]], dtype=np.float32)
                    kps_nr = (kps_5.astype(np.float32) - kps_5.mean(axis=0, keepdims=True)) @ R.T
                    # Landmarks indices: 0=left eye,1=right eye,2=nose,3=mouthL,4=mouthR
                    eye_cy = float((kps_nr[0, 1] + kps_nr[1, 1]) * 0.5)
                    nose_y = float(kps_nr[2, 1])
                    mouth_cy = float((kps_nr[3, 1] + kps_nr[4, 1]) * 0.5)
                    denom = max(1e-5, abs(nose_y - eye_cy))
                    r = (mouth_cy - eye_cy) / denom
                    # Baseline from template (dst) in its local coords
                    try:
                        dst_tmp = faceutil.get_arcface_template(image_size=512, mode='arcface128')
                        dst_tmp = np.squeeze(dst_tmp).astype(np.float32)
                        eye_cy0 = float((dst_tmp[0, 1] + dst_tmp[1, 1]) * 0.5)
                        nose_y0 = float(dst_tmp[2, 1])
                        mouth_cy0 = float((dst_tmp[3, 1] + dst_tmp[4, 1]) * 0.5)
                        denom0 = max(1e-5, abs(nose_y0 - eye_cy0))
                        r0 = (mouth_cy0 - eye_cy0) / denom0
                    except Exception:
                        r0 = 1.2  # safe fallback baseline
                    delta = r - r0
                    direction = 1.0 if delta > 0 else (-1.0 if delta < 0 else 0.0)
                    magnitude = min(1.0, abs(delta) / max(1e-5, r0))
                    eff = float(strength_ui * min(1.0, magnitude * 2.0))
                    if direction != 0.0 and eff > 0.0:
                        # Build vertical falloff in 128-space
                        H, W = 128, 128
                        y = torch.linspace(0, 1, H, device=self.models_processor.device, dtype=torch.float32).view(1, H, 1)
                        y = y.expand(1, H, W)
                        # smoothstep helper
                        def smoothstep(a: float, b: float, t: torch.Tensor):
                            tt = torch.clamp((t - a) / max(1e-5, (b - a)), 0.0, 1.0)
                            return tt * tt * (3.0 - 2.0 * tt)
                        if direction > 0:  # head down -> attenuate bottom
                            falloff = smoothstep(0.5, 1.0, y)
                        else:  # head up -> attenuate top
                            falloff = smoothstep(0.5, 1.0, 1.0 - y)
                        atten = torch.clamp(1.0 - eff * falloff, 0.0, 1.0)
                        swap_mask = swap_mask * atten
        except Exception:
            pass

        # Add blur to swap_mask results
        gauss = transforms.GaussianBlur(parameters['OverallMaskBlendAmountSlider'] * 2 + 1, (parameters['OverallMaskBlendAmountSlider'] + 1) * 0.2)
        swap_mask = gauss(swap_mask)
        # Optional adaptive feather proportional to the mask size (128-base -> scales with face)
        try:
            if parameters.get('AdaptiveFeatherEnableToggle', False) or (ai_tune_on_local and ai_feather_pct_local > 0.0):
                pct = float(parameters.get('AdaptiveFeatherPercentSlider', 0))
                if ai_tune_on_local:
                    pct = max(pct, ai_feather_pct_local)
                if pct > 0:
                    # Map percent to a reasonable radius in 128px space (cap to avoid huge kernels)
                    radius = int(max(1, min(32, round((pct / 100.0) * 24))))
                    gauss_af = transforms.GaussianBlur(radius * 2 + 1, (radius + 1) * 0.2)
                    swap_mask = gauss_af(swap_mask)
        except Exception:
            pass

        # Optional: Localize swap impact near face center to keep effect tight
        try:
            if parameters.get('SimilarityLocalityEnableToggle', False):
                # Build a radial mask in 128 space centered at (64,64)
                Hc, Wc = 128, 128
                yy, xx = torch.meshgrid(torch.arange(Hc, device=swap_mask.device), torch.arange(Wc, device=swap_mask.device), indexing='ij')
                cx, cy = Wc * 0.5, Hc * 0.5
                # Radius from slider (% of crop size)
                r_pct = float(parameters.get('SimilarityLocalityRadiusPercentSlider', 35))
                r = max(1.0, min(100.0, r_pct)) * 0.01 * min(Wc, Hc) * 0.5
                # Feather width as percent of radius
                f_pct = float(parameters.get('SimilarityLocalityFeatherPercentSlider', 30))
                f = max(0.0, min(100.0, f_pct)) * 0.01 * r
                dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                # Smooth step: 1 inside radius, fade to 0 across feather band
                inner = r
                outer = r + max(1e-3, f)
                t = (dist - inner) / (outer - inner)
                t = torch.clamp(t, 0.0, 1.0)
                radial = 1.0 - t
                radial = radial.unsqueeze(0)  # (1,128,128)
                # Apply to swap mask (already in 128 space here)
                swap_mask = torch.mul(swap_mask, radial)
        except Exception:
            pass

        # Combine border and swap mask
        swap_mask = torch.mul(swap_mask, border_mask)
        # Upscale the mask to face space (512)
        swap_mask = t512(swap_mask)

        # New: Edge-aware adaptive smoothing on the upscaled mask guided by the face content
        try:
            if parameters.get('EdgeSmoothingEnableToggle', False) or (ai_tune_on_local and ai_edge_strength_local > 0.0):
                strength_ui = float(parameters.get('EdgeSmoothingStrengthSlider', 0))
                if ai_tune_on_local:
                    # Transient AI tuning may raise the minimum strength
                    strength_ui = max(strength_ui, float(ai_edge_strength_local))
                # Normalize to [0,1]
                s_norm = max(0.0, min(1.0, strength_ui / 100.0))
                if s_norm > 0.0:
                    # Face guidance image in 512-space; mask shape (1,H,W)
                    try:
                        swap_mask = self.models_processor.face_masks.adaptive_border_smooth(
                            mask=swap_mask, image=original_face_512, strength=s_norm
                        )
                    except Exception:
                        # Best-effort: if guided smoothing fails, continue without it
                        pass
        except Exception:
            pass

        # Apply mask onto the swapped face content in face space
        swap = torch.mul(swap, swap_mask)

        # For face comparing
        original_face_512_clone = None
        if self.is_view_face_compare:
            original_face_512_clone = original_face_512.clone()
            original_face_512_clone = original_face_512_clone.type(torch.uint8)
            original_face_512_clone = original_face_512_clone.permute(1, 2, 0)
        swap_mask_clone = None
        # Uninvert and create image from swap mask
        if self.is_view_face_mask:
            swap_mask_clone = swap_mask.clone()
            swap_mask_clone = torch.sub(1, swap_mask_clone)
            swap_mask_clone = torch.cat((swap_mask_clone,swap_mask_clone,swap_mask_clone),0)
            swap_mask_clone = swap_mask_clone.permute(1, 2, 0)
            swap_mask_clone = torch.mul(swap_mask_clone, 255.).type(torch.uint8)

        # Calculate the area to be mergerd back to the original frame
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])

        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])

        left = floor(np.min(x))
        if left<0:
            left=0
        top = floor(np.min(y))
        if top<0:
            top=0
        right = ceil(np.max(x))
        if right>img.shape[2]:
            right=img.shape[2]
        bottom = ceil(np.max(y))
        if bottom>img.shape[1]:
            bottom=img.shape[1]

        # Paste-back: choose between classic inverse affine or non-rigid (piecewise) warp
        use_nonrigid = False
        try:
            if bool(control.get('NonRigidWarpEnableToggle', False)) and isinstance(kps_all, np.ndarray) and kps_all.ndim == 2 and kps_all.shape[1] == 2 and kps_all.shape[0] >= 32:
                mode = str(control.get('NonRigidWarpModeSelection', 'Auto'))
                if mode == 'Manual':
                    use_nonrigid = True
                else:
                    # Auto mode: require low-to-moderate motion and a minimum face size
                    sm = getattr(self.video_processor, 'scene_metrics', {}) or {}
                    motion = float(sm.get('motion', 0.0))
                    face_ratio = float(sm.get('face_ratio', 0.0))
                    # thresholds tuned for stability: motion <= 1.2 (low), face_ratio >= 0.08 (>=8% of frame height)
                    if motion <= 1.2 and face_ratio >= 0.08:
                        use_nonrigid = True
        except Exception:
            use_nonrigid = False

        if use_nonrigid:
            try:
                # Prepare control points: optionally downsample landmarks by 'Warp Detail'
                kp_img = kps_all.astype(np.float32)
                # Determine sampling stride from detail slider (1..100) -> stride in [1..6]
                try:
                    detail = int(control.get('NonRigidWarpDetailSlider', 60))
                except Exception:
                    detail = 60
                detail = max(1, min(100, detail))
                # Higher detail -> smaller stride
                stride = int(round(np.interp(100 - detail, [0, 100], [1, 6])))
                if stride > 1:
                    N = kp_img.shape[0]
                    idx = np.arange(N)
                    sel = (idx % stride == 0)
                    # Always keep priority facial regions when available
                    try:
                        pri = getattr(self.models_processor, 'LandmarksSubsetIdxs', [])
                        if isinstance(pri, (list, tuple, np.ndarray)):
                            pri_idx = np.array([int(i) for i in pri if isinstance(i, (int, np.integer)) and 0 <= int(i) < N], dtype=np.int32)
                            sel[pri_idx] = True
                    except Exception:
                        pass
                    # Ensure enough control points remain; if too few, relax selection
                    if sel.sum() < 16:
                        sel = np.ones(N, dtype=bool)
                    kp_img = kp_img[sel]

                # Map image landmarks into crop (512) space via forward tform
                M = tform.params[0:2]
                ones = np.ones((kp_img.shape[0], 1), dtype=np.float32)
                hom = np.hstack([kp_img, ones])
                src_pts_crop = (hom @ M.T)  # (N,2) in 512-crop coords
                # Destination points in image ROI local coords
                dst_pts_img = kp_img
                dst_pts_local = dst_pts_img.copy()
                dst_pts_local[:, 0] -= float(left)
                dst_pts_local[:, 1] -= float(top)
                # Filter points to valid domains
                def _in_bounds(pts, w, h):
                    return (pts[:, 0] >= 0) & (pts[:, 0] <= (w - 1)) & (pts[:, 1] >= 0) & (pts[:, 1] <= (h - 1))
                h_roi = int(max(0, bottom - top)); w_roi = int(max(0, right - left))
                mask_src = _in_bounds(src_pts_crop, 512, 512)
                mask_dst = _in_bounds(dst_pts_local, w_roi, h_roi)
                m = mask_src & mask_dst
                if m.sum() >= 16 and h_roi > 1 and w_roi > 1:
                    src = src_pts_crop[m]
                    dst = dst_pts_local[m]
                    # Build piecewise affine transform and warp both swap and mask to ROI
                    pw = trans.PiecewiseAffineTransform()
                    pw.estimate(src, dst)
                    # Prepare numpy images
                    swap_np = swap.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) / 255.0  # (512,512,3)
                    mask_np = swap_mask.detach().cpu().numpy().astype(np.float32)  # (1,512,512)
                    mask_np = np.transpose(mask_np, (1, 2, 0))  # (512,512,1)
                    # Warp into ROI shape
                    warped_swap = trans.warp(swap_np, inverse_map=pw.inverse, output_shape=(h_roi, w_roi), order=1, mode='edge', preserve_range=True)
                    warped_mask = trans.warp(mask_np, inverse_map=pw.inverse, output_shape=(h_roi, w_roi), order=1, mode='edge', preserve_range=True)

                    # Also compute classic inverse affine as a regularization base
                    swap_aff = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
                    swap_aff = v2.functional.affine(swap_aff, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
                    swap_aff = swap_aff[0:3, top:bottom, left:right].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                    mask_aff = v2.functional.pad(_edge_mask.permute(2,0,1), (0,0,img.shape[2]-512, img.shape[1]-512))
                    mask_aff = v2.functional.affine(mask_aff, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
                    mask_aff = mask_aff[0:1, top:bottom, left:right].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

                    # Blend with regularization percentage
                    reg = float(control.get('NonRigidWarpRegularizationSlider', 20)) / 100.0
                    reg = max(0.0, min(1.0, reg))
                    warped_swap = (1.0 - reg) * warped_swap + reg * (swap_aff.astype(np.float32) / 255.0)
                    warped_mask = (1.0 - reg) * warped_mask + reg * mask_aff

                    # Convert back to torch
                    swap = torch.from_numpy((warped_swap * 255.0).clip(0, 255).astype(np.uint8)).to(self.models_processor.device)
                    swap_mask = torch.from_numpy(np.clip(warped_mask, 0.0, 1.0).astype(np.float32)).to(self.models_processor.device)
                    # Ensure shapes: swap (H,W,3), mask (H,W,1)
                    if swap.dim() == 2:
                        swap = swap.unsqueeze(-1).repeat(1, 1, 3)
                    if swap_mask.dim() == 2:
                        swap_mask = swap_mask[..., None]
                else:
                    # Fallback to classic inverse affine if insufficient control points
                    raise RuntimeError('insufficient control points for non-rigid warp')
            except Exception:
                # Classic inverse affine (previous behavior)
                swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
                swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
                swap = swap[0:3, top:bottom, left:right]
                swap = swap.permute(1, 2, 0)

                swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
                swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
                swap_mask = swap_mask[0:1, top:bottom, left:right]
                swap_mask = swap_mask.permute(1, 2, 0)
        else:
            # Classic inverse affine (previous behavior)
            swap = v2.functional.pad(swap, (0,0,img.shape[2]-512, img.shape[1]-512))
            swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0,interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
            swap = swap[0:3, top:bottom, left:right]
            swap = swap.permute(1, 2, 0)

            swap_mask = v2.functional.pad(swap_mask, (0,0,img.shape[2]-512, img.shape[1]-512))
            swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0) )
            swap_mask = swap_mask[0:1, top:bottom, left:right]
            swap_mask = swap_mask.permute(1, 2, 0)

        # Keep a copy of non-inverted mask for edge operations
        _edge_mask = swap_mask.clone()
        swap_mask = torch.sub(1, swap_mask)

        # Apply the mask to the original image areas
        img_crop = img[0:3, top:bottom, left:right]
        img_crop = img_crop.permute(1,2,0)
        img_crop = torch.mul(swap_mask,img_crop)
            
        # Edge smoothing around mask border (guided/bilateral-like)
        try:
            if parameters.get('EdgeSmoothingEnableToggle', False) or (ai_tune_on_local and ai_edge_strength_local > 0.0):
                strength = float(parameters.get('EdgeSmoothingStrengthSlider', 0))
                if ai_tune_on_local:
                    strength = max(strength, ai_edge_strength_local)
                if strength > 0:
                    # Build a thin ring around the edge using dilate - erode on CPU for small ROI
                    m_np = _edge_mask.detach().cpu().numpy().astype(np.float32)
                    # radius from strength (1..8)
                    r = max(1, min(8, int(round(strength / 12.5))))
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                    dil = cv2.dilate(m_np, kernel)
                    ero = cv2.erode(m_np, kernel)
                    ring = np.clip(dil - ero, 0.0, 1.0).astype(np.float32)
                    if ring.max() > 0:
                        # Bilateral filter the swap content only within the ring
                        swap_np = swap.detach().cpu().numpy().astype(np.uint8)
                        # sigmaColor/sigmaSpace scale with strength
                        sigma_color = max(5, int(round(20 + strength * 1.2)))
                        sigma_space = max(3, int(round(1 + r * 2)))
                        bf = cv2.bilateralFilter(swap_np, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
                        ring3 = np.repeat(ring, 3, axis=2)
                        blended = (swap_np.astype(np.float32) * (1.0 - ring3) + bf.astype(np.float32) * ring3)
                        swap = torch.from_numpy(blended.astype(np.float32)).to(self.models_processor.device)
        except Exception:
            pass

        #Add the cropped areas and place them back into the original image
        swap = torch.add(swap, img_crop)
        swap = swap.type(torch.uint8)
        swap = swap.permute(2,0,1)
        img[0:3, top:bottom, left:right] = swap

        return img, original_face_512_clone, swap_mask_clone, tform

    def enhance_core(self, img, control):
        enhancer_type = control['FrameEnhancerTypeSelection']

        match enhancer_type:
            case 'RealEsrgan-x2-Plus' | 'RealEsrgan-x4-Plus' | 'BSRGan-x2' | 'BSRGan-x4' | 'UltraSharp-x4' | 'UltraMix-x4' | 'RealEsr-General-x4v3':
                tile_size = 512

                if enhancer_type == 'RealEsrgan-x2-Plus' or enhancer_type == 'BSRGan-x2':
                    scale = 2
                else:
                    scale = 4

                image = img.type(torch.float32)
                if torch.max(image) > 256:  # 16-bit image
                    max_range = 65535
                else:
                    max_range = 255

                image = torch.div(image, max_range)
                image = torch.unsqueeze(image, 0).contiguous()

                image = self.models_processor.run_enhance_frame_tile_process(image, enhancer_type, tile_size=tile_size, scale=scale)

                image = torch.squeeze(image)
                image = torch.clamp(image, 0, 1)
                image = torch.mul(image, max_range)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"])/100.0

                t_scale = v2.Resize((img.shape[1] * scale, img.shape[2] * scale), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                img = t_scale(img)
                img = torch.add(torch.mul(image, alpha), torch.mul(img, 1-alpha))
                if max_range == 255:
                    img = img.type(torch.uint8)
                else:
                    img = img.type(torch.uint16)

            case 'DeOldify-Artistic' | 'DeOldify-Stable' | 'DeOldify-Video':
                render_factor = 384 # 12 * 32 | highest quality = 20 * 32 == 640

                _, h, w = img.shape
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)

                image = image.type(torch.float32)
                image = torch.unsqueeze(image, 0).contiguous()

                output = torch.empty((image.shape), dtype=torch.float32, device=self.models_processor.device).contiguous()

                match enhancer_type:
                    case 'DeOldify-Artistic':
                        self.models_processor.run_deoldify_artistic(image, output)
                    case 'DeOldify-Stable':
                        self.models_processor.run_deoldify_stable(image, output)
                    case 'DeOldify-Video':
                        self.models_processor.run_deoldify_video(image, output)

                output = torch.squeeze(output)
                t_resize_o = v2.Resize((h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output = t_resize_o(output)

                output = faceutil.rgb_to_yuv(output, True)
                # do a black and white transform first to get better luminance values
                hires = faceutil.rgb_to_yuv(img, True)

                hires[1:3, :, :] = output[1:3, :, :]
                hires = faceutil.yuv_to_rgb(hires, True)

                # Blend
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                img = torch.add(torch.mul(hires, alpha), torch.mul(img, 1-alpha))

                img = img.type(torch.uint8)

            case 'DDColor-Artistic' | 'DDColor':
                render_factor = 384 # 12 * 32 | highest quality = 20 * 32 == 640

                # Converti RGB a LAB
                #'''
                #orig_l = img.permute(1, 2, 0).cpu().numpy()
                #orig_l = cv2.cvtColor(orig_l, cv2.COLOR_RGB2Lab)
                #orig_l = torch.from_numpy(orig_l).to(self.models_processor.device)
                #orig_l = orig_l.permute(2, 0, 1)
                #'''
                orig_l = faceutil.rgb_to_lab(img, True)

                orig_l = orig_l[0:1, :, :]  # (1, h, w)

                # Resize per il modello
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)

                # Converti RGB in LAB
                #'''
                #img_l = image.permute(1, 2, 0).cpu().numpy()
                #img_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2Lab)
                #img_l = torch.from_numpy(img_l).to(self.models_processor.device)
                #img_l = img_l.permute(2, 0, 1)
                #'''
                img_l = faceutil.rgb_to_lab(image, True)

                img_l = img_l[0:1, :, :]  # (1, render_factor, render_factor)
                img_gray_lab = torch.cat((img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)), dim=0)  # (3, render_factor, render_factor)

                # Converti LAB in RGB
                #'''
                #img_gray_lab = img_gray_lab.permute(1, 2, 0).cpu().numpy()
                #img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
                #img_gray_rgb = torch.from_numpy(img_gray_rgb).to(self.models_processor.device)
                #img_gray_rgb = img_gray_rgb.permute(2, 0, 1)
                #'''
                img_gray_rgb = faceutil.lab_to_rgb(img_gray_lab)

                tensor_gray_rgb = torch.unsqueeze(img_gray_rgb.type(torch.float32), 0).contiguous()

                # Prepara il tensore per il modello
                output_ab = torch.empty((1, 2, render_factor, render_factor), dtype=torch.float32, device=self.models_processor.device)

                # Esegui il modello
                match enhancer_type:
                    case 'DDColor-Artistic':
                        self.models_processor.run_ddcolor_artistic(tensor_gray_rgb, output_ab)
                    case 'DDColor':
                        self.models_processor.run_ddcolor(tensor_gray_rgb, output_ab)

                output_ab = output_ab.squeeze(0)  # (2, render_factor, render_factor)

                t_resize_o = v2.Resize((img.size(1), img.size(2)), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output_lab_resize = t_resize_o(output_ab)

                # Combina il canale L originale con il risultato del modello
                output_lab = torch.cat((orig_l, output_lab_resize), dim=0)  # (3, original_H, original_W)

                # Convert LAB to RGB
                #'''
                #output_rgb = output_lab.permute(1, 2, 0).cpu().numpy()
                #output_rgb = cv2.cvtColor(output_rgb, cv2.COLOR_Lab2RGB)
                #output_rgb = torch.from_numpy(output_rgb).to(self.models_processor.device)
                #output_rgb = output_rgb.permute(2, 0, 1)
                #'''
                output_rgb = faceutil.lab_to_rgb(output_lab, True)  # (3, original_H, original_W)

                # Miscela le immagini
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                blended_img = torch.add(torch.mul(output_rgb, alpha), torch.mul(img, 1 - alpha))

                # Converti in uint8
                img = blended_img.type(torch.uint8)

        return img

    def apply_face_expression_restorer(self, driving, target, parameters):
        """ Apply face expression restorer from driving to target.

        Args:
        driving (torch.Tensor: uint8): Driving image tensor (C x H x W)
        target (torch.Tensor: float32): Target image tensor (C x H x W)
        parameters (dict).
        
        Returns:
        torch.Tensor (uint8 -> float32): Transformed image (C x H x W)
        """
        t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        #cv2.imwrite("driving.png", cv2.cvtColor(driving.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        _, driving_lmk_crop, _ = self.models_processor.run_detect_landmark(driving, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
        driving_face_512 = driving.clone()
        #cv2.imshow("driving", cv2.cvtColor(driving_face_512.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        driving_face_256 = t256(driving_face_512)

        # Making motion templates: driving_template_dct
        c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None]) #c_d_eyes_lst
        c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None]) #c_d_lip_lst
        x_d_i_info = self.models_processor.lp_motion_extractor(driving_face_256, 'Human-Face')
        R_d_i = faceutil.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
        ##
        
        # R_d_0, x_d_0_info = None, None
        driving_multiplier=parameters['FaceExpressionFriendlyFactorDecimalSlider'] # 1.0 # be used only when driving_option is "expression-friendly"
        animation_region = parameters['FaceExpressionAnimationRegionSelection'] # 'all' # lips, eyes, pose, exp

        flag_normalize_lip = parameters['FaceExpressionNormalizeLipsEnableToggle'] # True #inf_cfg.flag_normalize_lip  # not overwrite
        lip_normalize_threshold = parameters['FaceExpressionNormalizeLipsThresholdDecimalSlider'] # 0.03 # threshold for flag_normalize_lip
        flag_eye_retargeting = parameters['FaceExpressionRetargetingEyesEnableToggle'] # False #inf_cfg.flag_eye_retargeting
        eye_retargeting_multiplier = parameters['FaceExpressionRetargetingEyesMultiplierDecimalSlider']  # 1.00
        flag_lip_retargeting = parameters['FaceExpressionRetargetingLipsEnableToggle'] # False #inf_cfg.flag_lip_retargeting
        lip_retargeting_multiplier = parameters['FaceExpressionRetargetingLipsMultiplierDecimalSlider'] # 1.00
        
        # fix:
        if animation_region == 'all':
            animation_region = 'eyes,lips'

        flag_relative_motion = True #inf_cfg.flag_relative_motion
        flag_stitching = True #inf_cfg.flag_stitching
        flag_pasteback = True #inf_cfg.flag_pasteback
        flag_do_crop = True #inf_cfg.flag_do_crop
        
        lip_delta_before_animation, eye_delta_before_animation = None, None

        target = torch.clamp(target, 0, 255).type(torch.uint8)
        #cv2.imwrite("target.png", cv2.cvtColor(target.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        _, source_lmk, _ = self.models_processor.run_detect_landmark(target, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
        target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(target, source_lmk, dsize=512, scale=parameters['FaceExpressionCropScaleDecimalSlider'], vy_ratio=parameters['FaceExpressionVYRatioDecimalSlider'], interpolation=v2.InterpolationMode.BILINEAR)
        #cv2.imshow("target", cv2.cvtColor(target_face_512.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        target_face_256 = t256(target_face_512)

        x_s_info = self.models_processor.lp_motion_extractor(target_face_256, 'Human-Face')
        x_c_s = x_s_info['kp']
        R_s = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.models_processor.lp_appearance_feature_extractor(target_face_256, 'Human-Face')
        x_s = faceutil.transform_keypoint(x_s_info)

        # let lip-open scalar to be 0 at first
        if flag_normalize_lip and flag_relative_motion and source_lmk is not None:
            c_d_lip_before_animation = [0.]
            combined_lip_ratio_tensor_before_animation = faceutil.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk, device=self.models_processor.device)
            if combined_lip_ratio_tensor_before_animation[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

        #R_d_0 = R_d_i.clone()
        #x_d_0_info = x_d_i_info.copy()

        delta_new = x_s_info['exp'].clone()
        if flag_relative_motion:
            if animation_region == "all" or animation_region == "pose":
                #R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
                R_new = (R_d_i @ R_d_i.permute(0, 2, 1)) @ R_s
            else:
                R_new = R_s
            if animation_region == "all" or animation_region == "exp":
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device))
            else:
                if "lips" in animation_region:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device)))[:, lip_idx, :]

                if "eyes" in animation_region:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
            '''
            elif animation_region == "lips":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device)))[:, lip_idx, :]
            elif animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]
            '''
            if animation_region == "all":
                #scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
                scale_new = x_s_info['scale']
            else:
                scale_new = x_s_info['scale']
            if animation_region == "all" or animation_region == "pose":
                #t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
                t_new = x_s_info['t']
            else:
                t_new = x_s_info['t']
        else:
            if animation_region == "all" or animation_region == "pose":
                R_new = R_d_i
            else:
                R_new = R_s
            if animation_region == "all" or animation_region == "exp":
                for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                    delta_new[:, idx, :] = x_d_i_info['exp'][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_i_info['exp'][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_i_info['exp'][:, 5, 2]
                delta_new[:, 8, 2] = x_d_i_info['exp'][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_i_info['exp'][:, 9, 1:]
            else:
                if "lips" in animation_region:
                    for lip_idx in [6, 12, 14, 17, 19, 20]:
                        delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]

                if "eyes" in animation_region:
                    for eyes_idx in [11, 13, 15, 16, 18]:
                        delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
            '''
            elif animation_region == "lips":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_i_info['exp'][:, lip_idx, :]
            elif animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_d_i_info['exp'][:, eyes_idx, :]
            '''
            scale_new = x_s_info['scale']
            if animation_region == "all" or animation_region == "pose":
                t_new = x_d_i_info['t']
            else:
                t_new = x_s_info['t']

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
        
        if not flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
            # without stitching or retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new += lip_delta_before_animation

        elif flag_stitching and not flag_eye_retargeting and not flag_lip_retargeting:
            # with stitching and without retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"]) + lip_delta_before_animation
            else:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])

        else:
            eyes_delta, lip_delta = None, None
            if flag_eye_retargeting and source_lmk is not None:
                c_d_eyes_i = c_d_eyes_lst
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio(c_d_eyes_i, source_lmk, device=self.models_processor.device)
                combined_eye_ratio_tensor = combined_eye_ratio_tensor * eye_retargeting_multiplier
                # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)                
                eyes_delta = self.models_processor.lp_retarget_eye(x_s, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

            if flag_lip_retargeting and source_lmk is not None:
                c_d_lip_i = c_d_lip_lst
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio(c_d_lip_i, source_lmk, device=self.models_processor.device)
                combined_lip_ratio_tensor = combined_lip_ratio_tensor * lip_retargeting_multiplier
                # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                lip_delta = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])

            if flag_relative_motion:  # use x_s
                x_d_i_new = x_s + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)
            else:  # use x_d,i
                x_d_i_new = x_d_i_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            if flag_stitching:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])

        x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier

        out = self.models_processor.lp_warp_decode(f_s, x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])
        out = torch.squeeze(out)
        out = torch.clamp(out, 0, 1)  # Clip i valori tra 0 e 1

        # Applica la maschera
        #out = torch.mul(out, self.models_processor.lp_mask_crop)  # Applica la maschera

        if flag_pasteback and flag_do_crop and flag_stitching:
            t = trans.SimilarityTransform()
            t.params[0:2] = M_c2o
            dsize = (target.shape[1], target.shape[2])
            # pad image by image size
            out = faceutil.pad_image_by_size(out, dsize)
            out = v2.functional.affine(out, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=v2.InterpolationMode.BILINEAR, center=(0, 0))
            out = v2.functional.crop(out, 0,0, dsize[0], dsize[1]) # cols, rows

        out = torch.clamp(torch.mul(out, 255.0), 0, 255).type(torch.float32)
        #cv2.imshow("output", cv2.cvtColor(out.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        return out

    def swap_edit_face_core(self, img, kps, parameters, control, **kwargs): # img = RGB
        # Grab 512 face from image and create 256 and 128 copys
        if parameters['FaceEditorEnableToggle']:
            # Scaling Transforms
            t256 = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

            # initial eye_ratio and lip_ratio values
            init_source_eye_ratio = 0.0
            init_source_lip_ratio = 0.0

            _, lmk_crop, _ = self.models_processor.run_detect_landmark( img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True)
            source_eye_ratio = faceutil.calc_eye_close_ratio(lmk_crop[None])
            source_lip_ratio = faceutil.calc_lip_close_ratio(lmk_crop[None])
            init_source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
            init_source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)

            # prepare_retargeting_image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(img, lmk_crop, dsize=512, scale=parameters["FaceEditorCropScaleDecimalSlider"], vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=v2.InterpolationMode.BILINEAR)
            original_face_256 = t256(original_face_512)

            x_s_info = self.models_processor.lp_motion_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_d_info_user_pitch = x_s_info['pitch'] + parameters['HeadPitchSlider'] #input_head_pitch_variation
            x_d_info_user_yaw = x_s_info['yaw'] + parameters['HeadYawSlider'] # input_head_yaw_variation
            x_d_info_user_roll = x_s_info['roll'] + parameters['HeadRollSlider'] #input_head_roll_variation
            R_s_user = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d_user = faceutil.get_rotation_matrix(x_d_info_user_pitch, x_d_info_user_yaw, x_d_info_user_roll)
            f_s_user = self.models_processor.lp_appearance_feature_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_s_user = faceutil.transform_keypoint(x_s_info)

            #execute_image_retargeting
            mov_x = torch.tensor(parameters['XAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_y = torch.tensor(parameters['YAxisMovementDecimalSlider']).to(self.models_processor.device)
            mov_z = torch.tensor(parameters['ZAxisMovementDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_x = torch.tensor(parameters['EyeGazeHorizontalDecimalSlider']).to(self.models_processor.device)
            eyeball_direction_y = torch.tensor(parameters['EyeGazeVerticalDecimalSlider']).to(self.models_processor.device)
            smile = torch.tensor(parameters['MouthSmileDecimalSlider']).to(self.models_processor.device)
            wink = torch.tensor(parameters['EyeWinkDecimalSlider']).to(self.models_processor.device)
            eyebrow = torch.tensor(parameters['EyeBrowsDirectionDecimalSlider']).to(self.models_processor.device)
            lip_variation_zero = torch.tensor(parameters['MouthPoutingDecimalSlider']).to(self.models_processor.device)
            lip_variation_one = torch.tensor(parameters['MouthPursingDecimalSlider']).to(self.models_processor.device)
            lip_variation_two = torch.tensor(parameters['MouthGrinDecimalSlider']).to(self.models_processor.device)
            lip_variation_three = torch.tensor(parameters['LipsCloseOpenSlider']).to(self.models_processor.device)

            x_c_s = x_s_info['kp']
            delta_new = x_s_info['exp']
            scale_new = x_s_info['scale']
            t_new = x_s_info['t']
            R_d_new = (R_d_user @ R_s_user.permute(0, 2, 1)) @ R_s_user

            if eyeball_direction_x != 0 or eyeball_direction_y != 0:
                delta_new = faceutil.update_delta_new_eyeball_direction(eyeball_direction_x, eyeball_direction_y, delta_new)
            if smile != 0:
                delta_new = faceutil.update_delta_new_smile(smile, delta_new)
            if wink != 0:
                delta_new = faceutil.update_delta_new_wink(wink, delta_new)
            if eyebrow != 0:
                delta_new = faceutil.update_delta_new_eyebrow(eyebrow, delta_new)
            if lip_variation_zero != 0:
                delta_new = faceutil.update_delta_new_lip_variation_zero(lip_variation_zero, delta_new)
            if lip_variation_one !=  0:
                delta_new = faceutil.update_delta_new_lip_variation_one(lip_variation_one, delta_new)
            if lip_variation_two != 0:
                delta_new = faceutil.update_delta_new_lip_variation_two(lip_variation_two, delta_new)
            if lip_variation_three != 0:
                delta_new = faceutil.update_delta_new_lip_variation_three(lip_variation_three, delta_new)
            if mov_x != 0:
                delta_new = faceutil.update_delta_new_mov_x(-mov_x, delta_new)
            if mov_y !=0 :
                delta_new = faceutil.update_delta_new_mov_y(mov_y, delta_new)

            x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            eyes_delta, lip_delta = None, None

            input_eye_ratio = max(min(init_source_eye_ratio + parameters['EyesOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_eye_ratio != init_source_eye_ratio:
                combined_eye_ratio_tensor = faceutil.calc_combined_eye_ratio([[float(input_eye_ratio)]], lmk_crop, device=self.models_processor.device)
                eyes_delta = self.models_processor.lp_retarget_eye(x_s_user, combined_eye_ratio_tensor, parameters["FaceEditorTypeSelection"])

            input_lip_ratio = max(min(init_source_lip_ratio + parameters['LipsOpenRatioDecimalSlider'], 0.80), 0.00)
            if input_lip_ratio != init_source_lip_ratio:
                combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio([[float(input_lip_ratio)]], lmk_crop, device=self.models_processor.device)
                lip_delta = self.models_processor.lp_retarget_lip(x_s_user, combined_lip_ratio_tensor, parameters["FaceEditorTypeSelection"])

            x_d_new = x_d_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            flag_stitching_retargeting_input: bool = kwargs.get('flag_stitching_retargeting_input', True)
            if flag_stitching_retargeting_input:
                x_d_new = self.models_processor.lp_stitching(x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])

            out = self.models_processor.lp_warp_decode(f_s_user, x_s_user, x_d_new, parameters["FaceEditorTypeSelection"])
            out = torch.squeeze(out)
            out = torch.clamp(out, 0, 1)  # clip to 0~1

            flag_do_crop_input_retargeting_image = kwargs.get('flag_do_crop_input_retargeting_image', True)
            if flag_do_crop_input_retargeting_image:
                gauss = transforms.GaussianBlur(parameters['FaceEditorBlurAmountSlider']*2+1, (parameters['FaceEditorBlurAmountSlider']+1)*0.2)
                mask_crop = gauss(self.models_processor.lp_mask_crop)
                img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)
            else:
                img = out                
                img = torch.mul(img, 255.0)
                img = torch.clamp(img, 0, 255).type(torch.uint8)

        if parameters['FaceMakeupEnableToggle'] or parameters['HairMakeupEnableToggle'] or parameters['EyeBrowsMakeupEnableToggle'] or parameters['LipsMakeupEnableToggle']:
            _, lmk_crop, _ = self.models_processor.run_detect_landmark( img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True)

            # prepare_retargeting_image
            original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(img, lmk_crop, dsize=512, scale=parameters['FaceEditorCropScaleDecimalSlider'], vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=v2.InterpolationMode.BILINEAR)

            out, mask_out = self.models_processor.apply_face_makeup(original_face_512, parameters)
            if 1:
                gauss = transforms.GaussianBlur(5*2+1, (5+1)*0.2)
                out = torch.clamp(torch.div(out, 255.0), 0, 1).type(torch.float32)
                mask_crop = gauss(self.models_processor.lp_mask_crop)
                img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        return img
