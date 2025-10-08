import os
import sys
import time
import torch
import numpy as np

# Minimal stub for signals
class _Sig:
    def emit(self, *args, **kwargs):
        return None

class _MW:
    def __init__(self):
        self.model_loading_signal = _Sig()
        self.model_loaded_signal = _Sig()
        self.control = {'MaxDFMModelsSlider': 1}
        self.dfm_models_data = {}


def time_it(fn, iters=10):
    # Warmup one call to account for lazy init
    fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    return (t1 - t0) / iters


if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from app.processors.models_processor import ModelsProcessor

    mw = _MW()
    mp = ModelsProcessor(mw, device='cuda')
    try:
        mp.switch_providers_priority('Auto')
    except Exception:
        pass

    # FaceParser benchmark
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    parser_in = (torch.rand((1, 3, 512, 512), dtype=torch.float32) - mean) / std
    parser_out = torch.empty((1, 19, 512, 512), dtype=torch.float32, device='cpu')
    def _parser_run():
        mp.run_faceparser(parser_in, parser_out)
    t_parser = time_it(_parser_run, iters=10)

    # Landmarks 478 benchmark
    img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    bbox = np.array([100, 100, 400, 400], dtype=np.float32)
    def _lmk_run():
        mp.face_landmark_detectors.run_detect_landmark(img, bbox, det_kpss=np.array([]), detect_mode='478', score=0.0, from_points=False)
    t_lmk = time_it(_lmk_run, iters=10)

    # GhostFace v2 benchmark (latent + run)
    emb = torch.randn((1, 512), dtype=torch.float32)
    gf_in = torch.rand((1, 3, 256, 256), dtype=torch.float32)
    gf_out = torch.empty((1, 3, 256, 256), dtype=torch.float32, device='cpu')
    def _gf_run():
        mp.run_swapper_ghostface(gf_in, emb, gf_out, 'GhostFace-v2')
    t_gf = time_it(_gf_run, iters=5)

    print(f"FaceParser avg: {t_parser*1000:.1f} ms")
    print(f"Landmarks478 avg: {t_lmk*1000:.1f} ms")
    print(f"GhostFace-v2 avg: {t_gf*1000:.1f} ms")
