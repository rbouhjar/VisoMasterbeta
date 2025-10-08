from typing import TYPE_CHECKING, Optional

import torch
from torchvision.transforms import v2
import numpy as np

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor

from app.processors.utils import faceutil

class FaceDetectors:
    def __init__(self, models_processor: 'ModelsProcessor'):
        self.models_processor = models_processor
        # Track whether the optional MediaPipe package is available
        self._mediapipe_available: Optional[bool] = None
        # After first confirmed unavailability, force fallback and notify only once
        self._mediapipe_force_fallback: bool = False
        self._mediapipe_notified_once: bool = False
        # Proactively detect MediaPipe availability once at startup
        try:
            import os
            # Silence verbose logs from underlying mediapipe/libs if user hasn't set them
            if 'GLOG_minloglevel' not in os.environ:
                os.environ['GLOG_minloglevel'] = '2'  # 0=INFO,1=WARNING,2=ERROR
            if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            from importlib import import_module
            import_module('mediapipe')
            self._mediapipe_available = True
            try:
                import absl.logging as absl_log
                absl_log.set_verbosity(absl_log.ERROR)
            except Exception:
                pass
        except Exception:
            self._mediapipe_available = False

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _center_distance_pct(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx = 0.5 * (ax1 + ax2); acy = 0.5 * (ay1 + ay2)
        bcx = 0.5 * (bx1 + bx2); bcy = 0.5 * (by1 + by2)
        dx = acx - bcx; dy = acy - bcy
        # Normalize by average diagonal length to be scale-invariant
        aw = max(1.0, ax2 - ax1); ah = max(1.0, ay2 - ay1)
        bw = max(1.0, bx2 - bx1); bh = max(1.0, by2 - by1)
        diag = 0.5 * ((aw**2 + ah**2) ** 0.5 + (bw**2 + bh**2) ** 0.5)
        return float((dx*dx + dy*dy) ** 0.5 / diag * 100.0)

    @staticmethod
    def _bbox_from_points(pts: np.ndarray, img_hw: tuple, padding: float = 0.06) -> Optional[np.ndarray]:
        """Compute a padded bounding box [x1,y1,x2,y2] from landmark points.
        - pts: (N,2)
        - img_hw: (H,W)
        - padding: fraction of max(width,height) to add as margin.
        Returns None if pts invalid.
        """
        try:
            if pts is None:
                return None
            arr = np.asarray(pts, dtype=np.float32)
            if arr.size == 0:
                return None
            arr = arr.reshape(-1, 2)
            x1 = float(np.min(arr[:, 0])); y1 = float(np.min(arr[:, 1]))
            x2 = float(np.max(arr[:, 0])); y2 = float(np.max(arr[:, 1]))
            w = x2 - x1; h = y2 - y1
            pad = float(max(w, h) * max(0.0, padding))
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
            H, W = img_hw
            x1 = max(0.0, min(x1, W - 1.0))
            y1 = max(0.0, min(y1, H - 1.0))
            x2 = max(0.0, min(x2, W - 1.0))
            y2 = max(0.0, min(y2, H - 1.0))
            if x2 <= x1:
                x2 = min(W - 1.0, x1 + 2.0)
            if y2 <= y1:
                y2 = min(H - 1.0, y1 + 2.0)
            return np.array([x1, y1, x2, y2], dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def _nms(pre_det: np.ndarray, thresh: float = 0.4) -> np.ndarray:
        """Perform Non-Maximum Suppression on detections.
        pre_det: array of shape (N, 5) with columns [x1, y1, x2, y2, score].
        Returns indices of boxes to keep, relative to pre_det.
        """
        if pre_det is None or len(pre_det) == 0:
            return np.array([], dtype=int)
        dets = pre_det
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=int)

    def _get_nms_thresh(self, default: float = 0.4) -> float:
        """Fetch NMS threshold from UI control if present; fallback to default."""
        try:
            control = self.models_processor.main_window.control
            # Expect percent slider 0-100 (e.g., 40 => 0.4)
            val = control.get('DetectionNMSThresholdSlider', int(default * 100))
            return max(0.0, min(1.0, float(val) / 100.0))
        except Exception:
            return default

    @staticmethod
    def _select_top_k(det: np.ndarray, kpss: Optional[np.ndarray], img_h: int, img_w: int, max_num: int):
        """Select up to max_num boxes preferring larger and centered faces.
        det: (N,5) during selection step (score in last column) or (N,4) after; works either way.
        Returns (det_sel, kpss_sel).
        """
        if not isinstance(det, np.ndarray) or det.shape[0] <= 1 or max_num <= 0:
            return det, kpss
        if det.shape[0] <= max_num:
            return det, kpss
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        det_img_center = (img_h // 2, img_w // 2)
        offsets = np.vstack([
            (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
            (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
        ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
        values = area - offset_dist_squared * 2.0  # extra weight on centering
        bindex = np.argsort(values)[::-1][:max_num]
        det_sel = det[bindex, :]
        if kpss is not None:
            try:
                kpss_sel = kpss[bindex, :]
            except Exception:
                kpss_sel = kpss
        else:
            kpss_sel = kpss
        return det_sel, kpss_sel

    def _deduplicate(self, det: np.ndarray, kpss_5: np.ndarray, kpss: np.ndarray, scores: np.ndarray) -> tuple:
        """Merge near-identical detections based on IoU and center distance.
        Keeps the higher-score detection; removes the other. Returns filtered arrays.
        """
        try:
            control = self.models_processor.main_window.control
            if not bool(control.get('DetectionDeduplicateEnableToggle', True)):
                return det, kpss_5, kpss, scores
            iou_thr = float(control.get('DetectionDeduplicateIoUSlider', 55)) / 100.0
            center_thr = float(control.get('DetectionDeduplicateCenterPctSlider', 18))
        except Exception:
            iou_thr = 0.55; center_thr = 18.0

        n = det.shape[0]
        if n <= 1:
            return det, kpss_5, kpss, scores

        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i+1, n):
                if not keep[j]:
                    continue
                # det may contain a score column; use only [x1, y1, x2, y2]
                a = det[i, :4]; b = det[j, :4]
                iou = self._iou(a, b)

                # Center distance and size similarity
                cdist = self._center_distance_pct(a, b)
                aw = max(1.0, a[2] - a[0]); ah = max(1.0, a[3] - a[1])
                bw = max(1.0, b[2] - b[0]); bh = max(1.0, b[3] - b[1])
                area_a = aw * ah; area_b = bw * bh
                size_ratio = min(area_a, area_b) / max(area_a, area_b)

                # Containment: overlap relative to smaller box area
                inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
                inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
                iw = max(0.0, inter_x2 - inter_x1)
                ih = max(0.0, inter_y2 - inter_y1)
                inter = iw * ih
                contain_ratio = inter / max(1.0, min(area_a, area_b))

                # Optional: landmark similarity if available
                lm_close = False
                if isinstance(kpss_5, np.ndarray) and kpss_5.shape[0] == n:
                    try:
                        p = kpss_5[i]; q = kpss_5[j]
                        # Normalize mean point distance by avg diagonal
                        ap = np.asarray(p); bq = np.asarray(q)
                        if ap.ndim == 2 and bq.ndim == 2 and ap.shape == bq.shape:
                            # compute mean l2 distance
                            dists = np.linalg.norm(ap - bq, axis=1)
                            diag = 0.5 * (((aw**2 + ah**2) ** 0.5) + ((bw**2 + bh**2) ** 0.5))
                            if diag > 0:
                                mean_pct = float(np.mean(dists) / diag * 100.0)
                                lm_close = mean_pct <= 12.0
                    except Exception:
                        lm_close = False

                is_dup = (
                    (iou >= iou_thr) or
                    (contain_ratio >= 0.65) or
                    (cdist <= center_thr and size_ratio >= 0.6) or
                    lm_close
                )

                if is_dup:
                    if scores[i] >= scores[j]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        det_f = det[keep]
        kpss5_f = kpss_5[keep] if isinstance(kpss_5, np.ndarray) and kpss_5.shape[0] == n else kpss_5
        kpss_f = kpss[keep] if isinstance(kpss, np.ndarray) and kpss.shape[0] == n else kpss
        scores_f = scores[keep] if isinstance(scores, np.ndarray) and scores.shape[0] == n else scores
        return det_f, kpss5_f, kpss_f, scores_f

    def run_detect(self, img, detect_mode='RetinaFace', max_num=1, score=0.5, input_size=(512, 512), use_landmark_detection=False, landmark_detect_mode='203', landmark_score=0.5, from_points=False, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        bboxes = []
        kpss_5 = []
        kpss = []

        if detect_mode=='RetinaFace':
            if not self.models_processor.models['RetinaFace']:
                self.models_processor.models['RetinaFace'] = self.models_processor.load_model('RetinaFace')

            bboxes, kpss_5, kpss = self.detect_retinaface(img, max_num=max_num, score=score, input_size=input_size, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='SCRFD':
            if not self.models_processor.models['SCRFD2.5g']:
                self.models_processor.models['SCRFD2.5g'] = self.models_processor.load_model('SCRFD2.5g')

            bboxes, kpss_5, kpss = self.detect_scrdf(img, max_num=max_num, score=score, input_size=input_size, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yolov8':
            if not self.models_processor.models['YoloFace8n']:
                self.models_processor.models['YoloFace8n'] = self.models_processor.load_model('YoloFace8n')

            bboxes, kpss_5, kpss = self.detect_yoloface(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='Yunet':
            if not self.models_processor.models['YunetN']:
                self.models_processor.models['YunetN'] = self.models_processor.load_model('YunetN')

            bboxes, kpss_5, kpss = self.detect_yunet(img, max_num=max_num, score=score, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rotation_angles)

        elif detect_mode=='MediaPipe':
            # If we already know MediaPipe is unavailable or forced fallback, skip trying again
            if (getattr(self, '_mediapipe_available', None) is False) or getattr(self, '_mediapipe_force_fallback', False):
                if not self._mediapipe_notified_once:
                    try:
                        print('[Detector] MediaPipe not available; falling back to RetinaFace.')
                    except Exception:
                        pass
                    self._mediapipe_notified_once = True
                if not self.models_processor.models.get('RetinaFace'):
                    self.models_processor.models['RetinaFace'] = self.models_processor.load_model('RetinaFace')
                bboxes, kpss_5, kpss = self.detect_retinaface(
                    img,
                    max_num=max_num,
                    score=score,
                    input_size=input_size,
                    use_landmark_detection=use_landmark_detection,
                    landmark_detect_mode=landmark_detect_mode,
                    landmark_score=landmark_score,
                    from_points=from_points,
                    rotation_angles=rotation_angles
                )
            else:
                bboxes, kpss_5, kpss = self.detect_mediapipe(
                    img,
                    max_num=max_num,
                    score=score,
                    use_landmark_detection=use_landmark_detection,
                    landmark_detect_mode=landmark_detect_mode,
                    landmark_score=landmark_score,
                    from_points=from_points
                )
                # If MediaPipe isn't installed, set force fallback and notify once, then use RetinaFace
                if (len(bboxes) == 0) and (getattr(self, '_mediapipe_available', None) is False):
                    if not self._mediapipe_notified_once:
                        try:
                            print('[Detector] MediaPipe not available; falling back to RetinaFace.')
                        except Exception:
                            pass
                        self._mediapipe_notified_once = True
                    self._mediapipe_force_fallback = True
                    if not self.models_processor.models.get('RetinaFace'):
                        self.models_processor.models['RetinaFace'] = self.models_processor.load_model('RetinaFace')
                    bboxes, kpss_5, kpss = self.detect_retinaface(
                        img,
                        max_num=max_num,
                        score=score,
                        input_size=input_size,
                        use_landmark_detection=use_landmark_detection,
                        landmark_detect_mode=landmark_detect_mode,
                        landmark_score=landmark_score,
                        from_points=from_points,
                        rotation_angles=rotation_angles
                    )

        # Optional: filter out tiny faces (based on bbox diagonal vs image diagonal)
        try:
            control = self.models_processor.main_window.control
            min_on = bool(control.get('MinFaceSizeFilterEnableToggle', False))
            min_pct = float(control.get('MinFaceSizePercentSlider', 3.0))
        except Exception:
            min_on = False; min_pct = 3.0

        if min_on and isinstance(bboxes, np.ndarray) and bboxes.size > 0:
            try:
                # Determine image size from tensor (CHW)
                H = int(img.size()[1]); W = int(img.size()[2])
                img_diag = (H*H + W*W) ** 0.5
                dgs = np.sqrt(np.maximum(0.0, (bboxes[:,2]-bboxes[:,0])**2 + (bboxes[:,3]-bboxes[:,1])**2))
                keep = dgs >= (min_pct/100.0) * img_diag
                if np.any(~keep):
                    bboxes = bboxes[keep]
                    if isinstance(kpss_5, np.ndarray) and kpss_5.shape[0] == keep.shape[0]:
                        kpss_5 = kpss_5[keep]
                    if isinstance(kpss, np.ndarray) and kpss.shape[0] == keep.shape[0]:
                        kpss = kpss[keep]
            except Exception:
                pass

        # If nothing was found, perform auto-retry widening search space
        try:
            control = self.models_processor.main_window.control
            auto_retry = bool(control.get('DetectionAutoRetryEnableToggle', True))
        except Exception:
            auto_retry = True

        if auto_retry and ((not isinstance(bboxes, np.ndarray)) or bboxes.size == 0):
            # Build retry candidates: rotations, scores, input sizes
            if len(rotation_angles) > 1:
                rot_sets = [rotation_angles]
            else:
                rot_sets = [ [0, -10, 10], [0, -20, 20, -35, 35] ]
            score_steps = [float(score)]
            if float(score) > 0.3:
                score_steps.append(max(0.2, float(score) - 0.1))
            if 0.15 not in score_steps:
                score_steps.append(0.15)
            size_steps = [input_size]
            try:
                if isinstance(input_size, tuple) and max(input_size) < 640:
                    size_steps.append((640, 640))
            except Exception:
                pass

            def _try(mode, s, ins, rots):
                if mode=='RetinaFace':
                    return self.detect_retinaface(img, max_num=max_num, score=s, input_size=ins, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rots)
                if mode=='SCRFD':
                    return self.detect_scrdf(img, max_num=max_num, score=s, input_size=ins, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rots)
                if mode=='Yolov8':
                    return self.detect_yoloface(img, max_num=max_num, score=s, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rots)
                if mode=='Yunet':
                    return self.detect_yunet(img, max_num=max_num, score=s, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points, rotation_angles=rots)
                return self.detect_mediapipe(img, max_num=max_num, score=s, use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points)

            for s in score_steps:
                for rots in rot_sets:
                    if detect_mode in ('RetinaFace', 'SCRFD'):
                        for ins in size_steps:
                            bboxes, kpss_5, kpss = _try(detect_mode, s, ins, rots)
                            if isinstance(bboxes, np.ndarray) and bboxes.size > 0:
                                return bboxes, kpss_5, kpss
                    else:
                        bboxes, kpss_5, kpss = _try(detect_mode, s, input_size, rots)
                        if isinstance(bboxes, np.ndarray) and bboxes.size > 0:
                            return bboxes, kpss_5, kpss

            # Final fallback: try MediaPipe if available
            if detect_mode!='MediaPipe' and not self._mediapipe_force_fallback and self._mediapipe_available is not False:
                try:
                    bboxes, kpss_5, kpss = self.detect_mediapipe(img, max_num=max_num, score=min(score_steps), use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=landmark_score, from_points=from_points)
                except Exception:
                    bboxes, kpss_5, kpss = [], [], []
                if isinstance(bboxes, np.ndarray) and bboxes.size > 0:
                    return bboxes, kpss_5, kpss

        return bboxes, kpss_5, kpss

    def detect_retinaface(self, img, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        #det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.models_processor.models['RetinaFace'].io_binding()
            io_binding.bind_input(name='input.1', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            io_binding.bind_output('448', self.models_processor.device)
            io_binding.bind_output('471', self.models_processor.device)
            io_binding.bind_output('494', self.models_processor.device)
            io_binding.bind_output('451', self.models_processor.device)
            io_binding.bind_output('474', self.models_processor.device)
            io_binding.bind_output('497', self.models_processor.device)
            io_binding.bind_output('454', self.models_processor.device)
            io_binding.bind_output('477', self.models_processor.device)
            io_binding.bind_output('500', self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['RetinaFace'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride
                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self._nms(pre_det, thresh=self._get_nms_thresh(0.4))
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        # Select top max_num by size and centering
        det, kpss = self._select_top_k(det, kpss, img_height, img_width, max_num)

    # Deduplicate overlapping detections before dropping scores
        score_values = det[:, 4]
        det, kpss, _, score_values = self._deduplicate(det, kpss, kpss, score_values)
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            refined = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(
                    img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points
                )
                refined.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(refined, dtype=object)

            # Optional: adjust bbox from landmarks
            try:
                control = self.models_processor.main_window.control
                use_pts_bbox = bool(control.get('DetectFromPointsToggle', False))
            except Exception:
                use_pts_bbox = from_points
            if use_pts_bbox:
                H, W = img_height, img_width
                for i in range(det.shape[0]):
                    pts = kpss[i] if isinstance(kpss, np.ndarray) and i < len(kpss) and len(kpss[i])>0 else kpss_5[i]
                    bb = self._bbox_from_points(pts, (H, W))
                    if bb is not None:
                        det[i, :4] = bb

        return det, kpss_5, kpss

    def detect_scrdf(self, img, max_num, score, input_size, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to RGB and normalize
        #det_img = det_img[:, :, [2,1,0]]
        det_img = torch.sub(det_img, 127.5)
        det_img = torch.div(det_img, 128.0)
        det_img = det_img.permute(2, 0, 1) #3,128,128

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.models_processor.models['SCRFD2.5g'].get_inputs()[0].name
        outputs = self.models_processor.models['SCRFD2.5g'].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()

            io_binding = self.models_processor.models['SCRFD2.5g'].io_binding()
            io_binding.bind_input(name=input_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['SCRFD2.5g'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            input_height = aimg.shape[2]
            input_width = aimg.shape[3]

            fmc = 3
            center_cache = {}
            for idx, stride in enumerate([8, 16, 32]):
                scores = net_outs[idx]
                bbox_preds = net_outs[idx+fmc]
                bbox_preds = bbox_preds * stride

                kps_preds = net_outs[idx+fmc*2] * stride
                height = input_height // stride
                width = input_width // stride
                key = (height, width, stride)
                if key in center_cache:
                    anchor_centers = center_cache[key]
                else:
                    anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                    anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                    anchor_centers = np.stack([anchor_centers]*2, axis=1).reshape( (-1,2) )
                    if len(center_cache)<100:
                        center_cache[key] = anchor_centers

                pos_inds = np.where(scores>=score)[0]

                x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
                y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
                x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
                y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

                bboxes = np.stack([x1, y1, x2, y2], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                preds = []
                for i in range(0, kps_preds.shape[1], 2):
                    px = anchor_centers[:, i%2] + kps_preds[:, i]
                    py = anchor_centers[:, i%2+1] + kps_preds[:, i+1]

                    preds.append(px)
                    preds.append(py)
                kpss = np.stack(preds, axis=-1)
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self._nms(pre_det, thresh=self._get_nms_thresh(0.4))
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        # Select top max_num by size and centering
        det, kpss = self._select_top_k(det, kpss, img_height, img_width, max_num)

        # Deduplicate overlapping detections before dropping scores
        score_values = det[:, 4]
        det, kpss, _, score_values = self._deduplicate(det, kpss, kpss, score_values)
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            refined = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(
                    img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points
                )
                refined.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(refined, dtype=object)

            try:
                control = self.models_processor.main_window.control
                use_pts_bbox = bool(control.get('DetectFromPointsToggle', False))
            except Exception:
                use_pts_bbox = from_points
            if use_pts_bbox:
                H, W = img_height, img_width
                for i in range(det.shape[0]):
                    pts = kpss[i] if isinstance(kpss, np.ndarray) and i < len(kpss) and len(kpss[i])>0 else kpss_5[i]
                    bb = self._bbox_from_points(pts, (H, W))
                    if bb is not None:
                        det[i, :4] = bb

        return det, kpss_5, kpss

    def detect_yoloface(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        # model_ratio = float(input_size[1]) / input_size[0]
        model_ratio = 1.0
        if im_ratio>model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=True)
        img = resize(img)
        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        det_img = det_img.permute(2, 0, 1)

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = aimg.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                aimg = det_img.permute(1, 2, 0)
                aimg = torch.div(aimg, 255.0)
                aimg = aimg.permute(2, 0, 1)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
                IM = None

            io_binding = self.models_processor.models['YoloFace8n'].io_binding()
            io_binding.bind_input(name='images', device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())
            io_binding.bind_output('output0', self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['YoloFace8n'].run_with_iobinding(io_binding)

            net_outs = io_binding.copy_outputs_to_cpu()

            outputs = np.squeeze(net_outs).T

            bbox_raw, score_raw, kps_raw, *_ = np.split(outputs, [4, 5], axis=1)

            keep_indices = np.where(score_raw > score)[0]

            if keep_indices.any():
                bbox_raw, kps_raw, score_raw = bbox_raw[keep_indices], kps_raw[keep_indices], score_raw[keep_indices]

                # Compute the transformed bounding box coordinates
                x1 = bbox_raw[:, 0] - bbox_raw[:, 2] / 2
                y1 = bbox_raw[:, 1] - bbox_raw[:, 3] / 2
                x2 = bbox_raw[:, 0] + bbox_raw[:, 2] / 2
                y2 = bbox_raw[:, 1] + bbox_raw[:, 3] / 2

                # Stack the results into a single array
                bboxes_raw = np.stack((x1, y1, x2, y2), axis=-1)

                # bboxes
                if angle != 0:
                    if len(bboxes_raw) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = bboxes_raw[:, :2]  # (x1, y1)
                        points2 = bboxes_raw[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        bboxes_raw = np.hstack((points1, points2))

                kps_list = []
                for kps in kps_raw:
                    indexes = np.arange(0, len(kps), 3)
                    temp_kps = []
                    for index in indexes:
                        temp_kps.append([kps[index], kps[index + 1]])
                    kps_list.append(np.array(temp_kps))

                kpss_raw = np.stack(kps_list)

                if do_rotation:
                    for i in range(len(kpss_raw)):
                        face_size = max(bboxes_raw[i][2] - bboxes_raw[i][0], bboxes_raw[i][3] - bboxes_raw[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, kpss_raw[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            score_raw[i] = 0.0

                        if angle != 0:
                            kpss_raw[i] = faceutil.trans_points2d(kpss_raw[i], IM)

                    keep_indices = np.where(score_raw>=score)[0]
                    score_raw = score_raw[keep_indices]
                    bboxes_raw = bboxes_raw[keep_indices]
                    kpss_raw = kpss_raw[keep_indices]

                kpss_list.append(kpss_raw)
                bboxes_list.append(bboxes_raw)
                scores_list.append(score_raw)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self._nms(pre_det, thresh=self._get_nms_thresh(0.4))
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        # Select top max_num by size and centering
        det, kpss = self._select_top_k(det, kpss, img_height, img_width, max_num)

        # Deduplicate overlapping detections before dropping scores
        score_values = det[:, 4]
        det, kpss, _, score_values = self._deduplicate(det, kpss, kpss, score_values)
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            refined = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(
                    img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points
                )
                refined.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(refined, dtype=object)

            try:
                control = self.models_processor.main_window.control
                use_pts_bbox = bool(control.get('DetectFromPointsToggle', False))
            except Exception:
                use_pts_bbox = from_points
            if use_pts_bbox:
                H, W = img_height, img_width
                for i in range(det.shape[0]):
                    pts = kpss[i] if isinstance(kpss, np.ndarray) and i < len(kpss) and len(kpss[i])>0 else kpss_5[i]
                    bb = self._bbox_from_points(pts, (H, W))
                    if bb is not None:
                        det[i, :4] = bb

        return det, kpss_5, kpss

    def detect_yunet(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points, rotation_angles=None):
        rotation_angles = rotation_angles or [0]
        img_landmark = None
        if use_landmark_detection:
            img_landmark = img.clone()

        # Resize image to fit within the input_size
        input_size = (640, 640)
        img_height, img_width = (img.size()[1], img.size()[2])
        im_ratio = torch.div(img_height, img_width)

        model_ratio = 1.0
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = torch.div(new_height,  img.size()[1])

        resize = v2.Resize((new_height, new_width), antialias=False)
        img = resize(img)

        img = img.permute(1,2,0)

        det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.uint8, device=self.models_processor.device)
        det_img[:new_height,:new_width,  :] = img

        # Switch to BGR
        det_img = det_img[:, :, [2,1,0]]

        det_img = det_img.permute(2, 0, 1) #3,640,640

        scores_list = []
        bboxes_list = []
        kpss_list = []

        cx = input_size[0] / 2  # image center x coordinate
        cy = input_size[1] / 2  # image center y coordinate

        if len(rotation_angles) > 1:
            do_rotation = True
        else:
            do_rotation = False

        input_name = self.models_processor.models['YunetN'].get_inputs()[0].name
        outputs = self.models_processor.models['YunetN'].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        for angle in rotation_angles:
            # Prepare data and find model parameters
            if angle != 0:
                aimg, M = faceutil.transform(det_img, (cx, cy), 640, 1.0, angle)
                IM = faceutil.invertAffineTransform(M)
                aimg = torch.unsqueeze(aimg, 0).contiguous()
            else:
                IM = None
                aimg = torch.unsqueeze(det_img, 0).contiguous()
            aimg = aimg.to(dtype=torch.float32)

            io_binding = self.models_processor.models['YunetN'].io_binding()
            io_binding.bind_input(name=input_name, device_type=self.models_processor.device, device_id=0, element_type=np.float32,  shape=aimg.size(), buffer_ptr=aimg.data_ptr())

            for i in range(len(output_names)):
                io_binding.bind_output(output_names[i], self.models_processor.device)

            # Sync and run model
            if self.models_processor.device == "cuda":
                torch.cuda.synchronize()
            elif self.models_processor.device != "cpu":
                self.models_processor.syncvec.cpu()
            self.models_processor.models['YunetN'].run_with_iobinding(io_binding)
            net_outs = io_binding.copy_outputs_to_cpu()

            strides = [8, 16, 32]
            for idx, stride in enumerate(strides):
                cls_pred = net_outs[idx].reshape(-1, 1)
                obj_pred = net_outs[idx + len(strides)].reshape(-1, 1)
                reg_pred = net_outs[idx + len(strides) * 2].reshape(-1, 4)
                kps_pred = net_outs[idx + len(strides) * 3].reshape(
                    -1, 5 * 2)

                anchor_centers = np.stack(
                    np.mgrid[:(input_size[1] // stride), :(input_size[0] //
                                                            stride)][::-1],
                    axis=-1)
                anchor_centers = (anchor_centers * stride).astype(
                    np.float32).reshape(-1, 2)

                scores = (cls_pred * obj_pred)
                pos_inds = np.where(scores>=score)[0]

                bbox_cxy = reg_pred[:, :2] * stride + anchor_centers[:]
                bbox_wh = np.exp(reg_pred[:, 2:]) * stride
                tl_x = (bbox_cxy[:, 0] - bbox_wh[:, 0] / 2.)
                tl_y = (bbox_cxy[:, 1] - bbox_wh[:, 1] / 2.)
                br_x = (bbox_cxy[:, 0] + bbox_wh[:, 0] / 2.)
                br_y = (bbox_cxy[:, 1] + bbox_wh[:, 1] / 2.)

                bboxes = np.stack([tl_x, tl_y, br_x, br_y], axis=-1)

                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]

                # bboxes
                if angle != 0:
                    if len(pos_bboxes) > 0:
                        # Split the points into coordinates (x1, y1) and (x2, y2)
                        points1 = pos_bboxes[:, :2]  # (x1, y1)
                        points2 = pos_bboxes[:, 2:]  # (x2, y2)

                        # Apply the inverse of the rotation matrix to points1 and points2
                        points1 = faceutil.trans_points2d(points1, IM)
                        points2 = faceutil.trans_points2d(points2, IM)

                        _x1 = points1[:, 0]
                        _y1 = points1[:, 1]
                        _x2 = points2[:, 0]
                        _y2 = points2[:, 1]

                        if angle in (-270, 90):
                            # x1, y2, x2, y1
                            points1 = np.stack((_x1, _y2), axis=1)
                            points2 = np.stack((_x2, _y1), axis=1)
                        elif angle in (-180, 180):
                            # x2, y2, x1, y1
                            points1 = np.stack((_x2, _y2), axis=1)
                            points2 = np.stack((_x1, _y1), axis=1)
                        elif angle in (-90, 270):
                            # x2, y1, x1, y2
                            points1 = np.stack((_x2, _y1), axis=1)
                            points2 = np.stack((_x1, _y2), axis=1)

                        # Reassemble the transformed points into the format [x1', y1', x2', y2']
                        pos_bboxes = np.hstack((points1, points2))

                # kpss
                kpss = np.concatenate(
                    [((kps_pred[:, [2 * i, 2 * i + 1]] * stride) + anchor_centers)
                        for i in range(5)],
                    axis=-1)

                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]

                if do_rotation:
                    for i in range(len(pos_kpss)):
                        face_size = max(pos_bboxes[i][2] - pos_bboxes[i][0], pos_bboxes[i][3] - pos_bboxes[i][1])
                        angle_deg_to_front = faceutil.get_face_orientation(face_size, pos_kpss[i])
                        if angle_deg_to_front < -50.00 or angle_deg_to_front > 50.00:
                            pos_scores[i] = 0.0

                        if angle != 0:
                            pos_kpss[i] = faceutil.trans_points2d(pos_kpss[i], IM)

                    pos_inds = np.where(pos_scores>=score)[0]
                    pos_scores = pos_scores[pos_inds]
                    pos_bboxes = pos_bboxes[pos_inds]
                    pos_kpss = pos_kpss[pos_inds]

                kpss_list.append(pos_kpss)
                bboxes_list.append(pos_bboxes)
                scores_list.append(pos_scores)

        if len(bboxes_list) == 0:
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        det_scale = det_scale.numpy()###

        bboxes = np.vstack(bboxes_list) / det_scale

        kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        # NMS
        keep = self._nms(pre_det, thresh=self._get_nms_thresh(0.4))
        det = pre_det[keep, :]

        kpss = kpss[order, :, :]
        kpss = kpss[keep, :, :]

        # Select top max_num by size and centering
        det, kpss = self._select_top_k(det, kpss, img_height, img_width, max_num)

        # Prepare to deduplicate overlapping detections
        score_values = det[:, 4]
        det, kpss, _, score_values = self._deduplicate(det, kpss, kpss, score_values)
        det = np.delete(det, 4, 1)

        kpss_5 = kpss.copy()
        if use_landmark_detection and len(kpss_5) > 0:
            refined = []
            for i in range(kpss_5.shape[0]):
                landmark_kpss_5, landmark_kpss, landmark_scores = self.models_processor.run_detect_landmark(
                    img_landmark, det[i], kpss_5[i], landmark_detect_mode, landmark_score, from_points
                )
                refined.append(landmark_kpss if len(landmark_kpss) > 0 else kpss_5[i])
                if len(landmark_kpss_5) > 0:
                    if len(landmark_scores) > 0:
                        if np.mean(landmark_scores) > np.mean(score_values[i]):
                            kpss_5[i] = landmark_kpss_5
                    else:
                        kpss_5[i] = landmark_kpss_5
            kpss = np.array(refined, dtype=object)

        return det, kpss_5, kpss

    def detect_mediapipe(self, img, max_num, score, use_landmark_detection, landmark_detect_mode, landmark_score, from_points):
        """Optional: use MediaPipe face detection if installed. Returns (bboxes, kpss_5, kpss).
        If landmark detection is enabled with mode '478', use Face Mesh to return dense (~468) landmarks.
        Otherwise, use MediaPipe Face Detection and map available keypoints to 5 points.
        """
        try:
            from importlib import import_module
            mp = import_module('mediapipe')
            self._mediapipe_available = True
        except Exception:
            self._mediapipe_available = False
            return [], [], []

        # Convert CHW torch img to HWC numpy uint8
        try:
            img_np = img[0:3].type(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()
        except Exception:
            return [], [], []
        H, W = img_np.shape[:2]

        try:
            # If user requested dense landmarks (478), use Face Mesh (468 visible landmarks)
            if bool(use_landmark_detection) and str(landmark_detect_mode) in ("478", "468"):
                with mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=max(1, int(max_num)),
                    refine_landmarks=True,
                    min_detection_confidence=float(score),
                    min_tracking_confidence=float(landmark_score)
                ) as fm:
                    res = fm.process(img_np)
                    if not res.multi_face_landmarks:
                        # Try ROI-based Face Mesh using Face Detection to get candidate boxes
                        try:
                            with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=float(score)) as fd2:
                                res2 = fd2.process(img_np)
                                if res2 and res2.detections:
                                    dets = []
                                    kps5 = []
                                    kps_all = []
                                    left_eye_idxs = [33, 133, 159, 145]
                                    right_eye_idxs = [263, 362, 386, 374]
                                    idx_nose = 1
                                    idx_mouth_l = 61
                                    idx_mouth_r = 291
                                    pad = 0.2
                                    for d in res2.detections[:max(1, max_num)]:
                                        bb = d.location_data.relative_bounding_box
                                        x1 = float(max(0.0, bb.xmin * W)); y1 = float(max(0.0, bb.ymin * H))
                                        x2 = float(min(W - 1.0, (bb.xmin + bb.width) * W)); y2 = float(min(H - 1.0, (bb.ymin + bb.height) * H))
                                        # Pad ROI
                                        bw = x2 - x1; bh = y2 - y1
                                        cx = x1 + bw * 0.5; cy = y1 + bh * 0.5
                                        rw = bw * (1.0 + pad); rh = bh * (1.0 + pad)
                                        rx0 = int(max(0, np.floor(cx - rw * 0.5)))
                                        ry0 = int(max(0, np.floor(cy - rh * 0.5)))
                                        rx1 = int(min(W - 1, np.ceil(cx + rw * 0.5)))
                                        ry1 = int(min(H - 1, np.ceil(cy + rh * 0.5)))
                                        if rx1 <= rx0 + 5 or ry1 <= ry0 + 5:
                                            continue
                                        roi = img_np[ry0:ry1, rx0:rx1, :]
                                        res_roi = fm.process(roi)
                                        if not res_roi.multi_face_landmarks:
                                            continue
                                        f0 = res_roi.multi_face_landmarks[0]
                                        roi_h, roi_w = roi.shape[0], roi.shape[1]
                                        pts = np.array([[lm.x * roi_w + rx0, lm.y * roi_h + ry0] for lm in f0.landmark], dtype=np.float32)
                                        bb2 = self._bbox_from_points(pts, (H, W), padding=0.06)
                                        if bb2 is None:
                                            continue
                                        dets.append(bb2.tolist())
                                        left_center = np.mean(pts[left_eye_idxs, :], axis=0)
                                        right_center = np.mean(pts[right_eye_idxs, :], axis=0)
                                        if left_center[0] <= right_center[0]:
                                            eye_left, eye_right = left_center, right_center
                                        else:
                                            eye_left, eye_right = right_center, left_center
                                        pts5 = np.array([
                                            eye_left,
                                            eye_right,
                                            pts[idx_nose],
                                            pts[idx_mouth_l],
                                            pts[idx_mouth_r]
                                        ], dtype=np.float32)
                                        kps5.append(pts5)
                                        kps_all.append(pts)
                                    if len(dets) > 0:
                                        det = np.asarray(dets, dtype=np.float32)
                                        kpss_5 = np.asarray(kps5, dtype=np.float32)
                                        kpss = np.asarray(kps_all, dtype=np.float32)
                                        return det, kpss_5, kpss
                                # If ROI mesh also fails, final fallback to 5-point detection
                                if not res2 or not res2.detections:
                                    return [], [], []
                                dets_f = []
                                kps5_f = []
                                for d in res2.detections[:max(1, max_num)]:
                                    bb = d.location_data.relative_bounding_box
                                    x1 = max(0.0, bb.xmin * W)
                                    y1 = max(0.0, bb.ymin * H)
                                    x2 = min(W - 1.0, (bb.xmin + bb.width) * W)
                                    y2 = min(H - 1.0, (bb.ymin + bb.height) * H)
                                    dets_f.append([x1, y1, x2, y2])
                                    pts = []
                                    if d.location_data.relative_keypoints:
                                        for kp in d.location_data.relative_keypoints[:5]:
                                            pts.append([kp.x * W, kp.y * H])
                                    if len(pts) < 5:
                                        while len(pts) < 5:
                                            pts.append([0.0, 0.0])
                                    kps5_f.append(np.array(pts, dtype=np.float32))
                                det = np.asarray(dets_f, dtype=np.float32)
                                kpss_5 = np.asarray(kps5_f, dtype=np.float32)
                                return det, kpss_5, np.array([], dtype=np.float32)
                        except Exception:
                            return [], [], []
                    dets = []
                    kps5 = []
                    kps_all = []
                    # Indices for eye regions (use a small set and average to get stable centers)
                    left_eye_idxs = [33, 133, 159, 145]   # around left eye + eye corner
                    right_eye_idxs = [263, 362, 386, 374] # around right eye + eye corner
                    idx_nose = 1
                    idx_mouth_l = 61
                    idx_mouth_r = 291
                    for f in res.multi_face_landmarks[:max(1, max_num)]:
                        pts = np.array([[lm.x * W, lm.y * H] for lm in f.landmark], dtype=np.float32)  # (468,2)
                        # BBox from points with small padding
                        bb = self._bbox_from_points(pts, (H, W), padding=0.06)
                        if bb is None:
                            continue
                        dets.append(bb.tolist())
                        # Compute eye centers by averaging selected landmarks; then enforce left/right by x order
                        left_center = np.mean(pts[left_eye_idxs, :], axis=0)
                        right_center = np.mean(pts[right_eye_idxs, :], axis=0)
                        # Ensure correct ordering: index 0 is left (smaller x), 1 is right (larger x)
                        if left_center[0] <= right_center[0]:
                            eye_left, eye_right = left_center, right_center
                        else:
                            eye_left, eye_right = right_center, left_center
                        pts5 = np.array([
                            eye_left,
                            eye_right,
                            pts[idx_nose],
                            pts[idx_mouth_l],
                            pts[idx_mouth_r]
                        ], dtype=np.float32)
                        kps5.append(pts5)
                        kps_all.append(pts)
                    if len(dets) == 0:
                        return [], [], []
                    det = np.asarray(dets, dtype=np.float32)
                    kpss_5 = np.asarray(kps5, dtype=np.float32)
                    kpss = np.asarray(kps_all, dtype=np.float32)
                    return det, kpss_5, kpss
            else:
                # Fallback to MediaPipe Face Detection (5-ish points)
                with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=float(score)) as fd:
                    res = fd.process(img_np)
                    if not res.detections:
                        return [], [], []
                    dets = []
                    kps5 = []
                    for d in res.detections[:max(1, max_num)]:
                        bb = d.location_data.relative_bounding_box
                        x1 = max(0.0, bb.xmin * W)
                        y1 = max(0.0, bb.ymin * H)
                        x2 = min(W - 1.0, (bb.xmin + bb.width) * W)
                        y2 = min(H - 1.0, (bb.ymin + bb.height) * H)
                        dets.append([x1, y1, x2, y2])
                        pts = []
                        if d.location_data.relative_keypoints:
                            for kp in d.location_data.relative_keypoints[:5]:
                                pts.append([kp.x * W, kp.y * H])
                        if len(pts) < 5:
                            while len(pts) < 5:
                                pts.append([0.0, 0.0])
                        kps5.append(np.array(pts, dtype=np.float32))
                    det = np.array(dets, dtype=np.float32)
                    # Ensure consistent numeric dtype and shape (N,5,2)
                    try:
                        kpss_5 = np.asarray(kps5, dtype=np.float32)
                    except Exception:
                        kpss_5 = np.array(kps5, dtype=object)
                    return det, kpss_5, np.array([], dtype=np.float32)
        except Exception:
            return [], [], []