import os
import sys

print("PY:", sys.version)
print("EXE:", sys.executable)

# Show first few PATH entries for debugging
path_parts = os.environ.get("PATH", "").split(";")
print("PATH head:", ";".join(path_parts[:3]))

# Check PyYAML
try:
    import yaml  # type: ignore
    print("yaml OK:", getattr(yaml, "__version__", "unknown"))
except Exception as e:
    print("yaml import FAILED:", e)

# Check ONNX Runtime
try:
    import onnxruntime as ort
    print("ORT version:", ort.__version__)
    print("Providers:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime import FAILED:", e)

# Optional: PyTorch CUDA availability
try:
    import torch
    print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
except Exception as e:
    print("torch import FAILED:", e)
