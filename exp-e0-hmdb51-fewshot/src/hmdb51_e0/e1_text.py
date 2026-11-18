"""Native CLIP text encoding with the exact frozen joint-space checkpoint."""
from pathlib import Path
import numpy as np

from .clip_features import FrozenCLIP
from .provenance import sha256_file


class FrozenTextCLIP(FrozenCLIP):
    def __init__(self,checkpoint: Path,device: str='cuda'):
        super().__init__(checkpoint,device=device,batch_size=16)
        import clip.simple_tokenizer
        tokenizer=Path(clip.simple_tokenizer.__file__)
        self.identity={**self.identity,'text_encoder':'native_openai_clip',
                       'tokenizer':dict(context_length=77,truncate=False,
                                        bpe_sha256=sha256_file(Path(clip.simple_tokenizer.default_bpe())),
                                        tokenizer_source_sha256=sha256_file(tokenizer))}

    def encode_text(self,prompts: list[str]) -> np.ndarray:
        import clip
        if not prompts or any(not isinstance(p,str) or not p.strip() for p in prompts):
            raise ValueError('text prompts must be nonempty strings')
        self.verify_mode()
        result=[]
        with self.torch.inference_mode():
            for start in range(0,len(prompts),self.batch_size):
                try:
                    tokens=clip.tokenize(prompts[start:start+self.batch_size],context_length=77,truncate=False)
                except RuntimeError as exc:
                    raise ValueError('concept text exceeds CLIP context; truncation is forbidden') from exc
                result.append(self.model.encode_text(tokens.to(self.device)).float().cpu().numpy())
        values=np.concatenate(result).astype(np.float64)
        norms=np.linalg.norm(values,axis=1,keepdims=True)
        if values.shape!=(len(prompts),self.dimension) or not np.isfinite(values).all() or np.any(norms<=1e-6):
            raise ValueError('invalid native text embeddings')
        self.verify_frozen()
        return (values/norms).astype(np.float32)
