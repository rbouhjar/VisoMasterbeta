import sys
import os
import numpy as np
import torch

# Dummy Qt-like signal
class _Sig:
    def emit(self, *args, **kwargs):
        return None

# Minimal MainWindow stub
class _MW:
    def __init__(self):
        self.model_loading_signal = _Sig()
        self.model_loaded_signal = _Sig()
        self.control = {'MaxDFMModelsSlider': 1}
        self.dfm_models_data = {}


def main():
    # Ensure repo root is importable without needing PYTHONPATH
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from app.processors.models_processor import ModelsProcessor

    mw = _MW()
    mp = ModelsProcessor(mw, device='cuda')

    # Prefer Auto: TRT -> CUDA -> CPU
    try:
        mp.switch_providers_priority('Auto')
    except Exception:
        # Older code path without 'Auto'
        mp.switch_providers_priority('TensorRT')

    # Create a dummy RGB image tensor (C,H,W) uint8
    img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    bbox = np.array([100, 100, 400, 400], dtype=np.float32)

    # Run 478 landmarks (with per-session device-safe bindings)
    try:
        kps5, kps_all, scores = mp.face_landmark_detectors.run_detect_landmark(img, bbox, det_kpss=np.array([]), detect_mode='478', score=0.0, from_points=False)
        print('Landmarks478: OK', (len(kps5) if hasattr(kps5, '__len__') else 0))
    except Exception as e:
        print('Landmarks478: FAIL', e)
        sys.exit(2)

    # Optional: run faceparser mask on dummy input to exercise mask path
    try:
        # Prepare normalized float32 input for the parser (NCHW, mean/std normalization)
        input_img = torch.rand((1, 3, 512, 512), dtype=torch.float32)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        input_img = (input_img - mean) / std
        output = torch.empty((1, 19, 512, 512), dtype=torch.float32, device='cpu')
        mp.run_faceparser(input_img, output)
        print('FaceParser: OK', tuple(output.shape))
    except Exception as e:
        print('FaceParser: FAIL', e)
        sys.exit(3)

    print('SMOKE: PASS')


if __name__ == '__main__':
    main()
