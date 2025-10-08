import os
import sys

# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from app.processors.models_processor import ModelsProcessor

class _Sig:
    def emit(self, *args, **kwargs):
        return None

class _MW:
    def __init__(self):
        self.model_loading_signal = _Sig()
        self.model_loaded_signal = _Sig()
        self.control = {'MaxDFMModelsSlider': 1}
        self.dfm_models_data = {}

if __name__ == '__main__':
    mw = _MW()
    mp = ModelsProcessor(mw, device='cuda')
    # Keep current provider selection; Auto is fine if supported
    try:
        mp.switch_providers_priority('Auto')
    except Exception:
        pass
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    results = mp.warmup(targets)
    for k, v in results.items():
        print(f"{k}: {v}")
