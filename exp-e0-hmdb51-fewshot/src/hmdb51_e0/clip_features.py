"""Native OpenAI CLIP image encoding, with no text or task labels."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import numpy as np

from .provenance import digest, sha256_file

CHECKPOINT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
CHECKPOINT_URL = f"https://openaipublic.azureedge.net/clip/models/{CHECKPOINT_SHA256}/ViT-B-32.pt"


def preprocessing_identity(preprocess) -> list[dict]:
    """Serialize native transform parameters without process-local addresses."""
    transforms = []
    for transform in preprocess.transforms:
        if callable(transform) and hasattr(transform, "__qualname__"):
            transforms.append({"function": f"{transform.__module__}.{transform.__qualname__}"})
            continue
        descriptor = {"type": f"{type(transform).__module__}.{type(transform).__qualname__}"}
        for attribute in ("size", "interpolation", "max_size", "antialias", "mean", "std", "inplace"):
            if hasattr(transform, attribute):
                value = getattr(transform, attribute)
                if attribute == "interpolation":
                    value = str(value)
                if isinstance(value, tuple):
                    value = list(value)
                descriptor[attribute] = value
        transforms.append(descriptor)
    return transforms


def pool_features(raw: np.ndarray, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 16 or raw.shape[1] < 1 or not np.isfinite(raw).all():
        raise ValueError("expected finite 16 x d frame features")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("zero or degenerate frame feature")
    z = (raw / norms).astype(np.float32)
    return z, pool_normalized(z, epsilon)


def pool_normalized(z: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    # Accumulate in float64: float32 mean-roundoff otherwise invents motion
    # for constant frames, which block normalization can amplify substantially.
    statistics = np.asarray(z, dtype=np.float64)
    mean = statistics.mean(axis=0)
    std = statistics.std(axis=0, ddof=0)
    vector = np.concatenate((mean / (np.linalg.norm(mean) + epsilon), std / (np.linalg.norm(std) + epsilon)))
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= epsilon:
        raise ValueError("zero or nonfinite video feature")
    return (vector / norm).astype(np.float32)


class FrozenCLIP:
    def __init__(self, checkpoint: Path, device: str = "cuda", batch_size: int = 16):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import clip
        if sha256_file(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("checkpoint hash is not the official OpenAI ViT-B/32 checkpoint")
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError(f"requested device unavailable: {device}")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.model, self.preprocess = clip.load(str(checkpoint), device=device, jit=False)
        self.model.float().eval().requires_grad_(False)
        self.dimension = int(self.model.visual.output_dim)
        self.before_hash = self.parameter_hash()
        clip_root = Path(clip.__file__).parent
        self.identity = {
            "implementation": "openai/CLIP", "clip_version": importlib.metadata.version("clip"),
            "clip_source_sha256": digest({p.name: sha256_file(p) for p in sorted(clip_root.glob("*.py"))}),
            "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_source": CHECKPOINT_URL,
            "architecture": "ViT-B/32", "dimension": self.dimension,
            "preprocess": preprocessing_identity(self.preprocess), "color": "RGB", "dtype": "float32",
            "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
            "pillow": importlib.metadata.version("Pillow"), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cuda": torch.version.cuda, "batch_size": batch_size,
            "deterministic_algorithms": True, "tf32": False,
            "parameter_sha256_before": self.before_hash,
        }

    def parameter_hash(self) -> str:
        h = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            h.update(name.encode())
            array = tensor.detach().cpu().contiguous().numpy()
            h.update(str(array.dtype).encode())
            h.update(str(array.shape).encode())
            h.update(array.tobytes())
        return h.hexdigest()

    def encode(self, frames: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        if len(frames) != 16:
            raise ValueError("encoder requires exactly 16 RGB frames")
        torch = self.torch
        self.verify_mode()
        result = []
        with torch.inference_mode():
            for offset in range(0, 16, self.batch_size):
                batch = torch.stack([self.preprocess(Image.fromarray(f)) for f in frames[offset:offset + self.batch_size]])
                encoded = self.model.encode_image(batch.to(self.device)).float().cpu().numpy()
                result.append(encoded)
        output = np.concatenate(result)
        if output.shape != (16, self.dimension):
            raise ValueError("checkpoint output dimension mismatch")
        return output

    def verify_mode(self) -> None:
        if any(module.training for module in self.model.modules()) or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise ValueError("CLIP must remain frozen, evaluation-only and gradient-free")

    def verify_frozen(self) -> dict:
        self.verify_mode()
        after = self.parameter_hash()
        if after != self.before_hash:
            raise ValueError("CLIP parameters changed during feature extraction")
        return {"before": self.before_hash, "after": after, "unchanged": True}
