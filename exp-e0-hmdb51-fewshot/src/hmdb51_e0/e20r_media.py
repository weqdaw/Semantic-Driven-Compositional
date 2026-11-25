"""Paired frozen-CLIP inputs; original pixels never become implied model scores."""
import numpy as np
from PIL import Image
from .clip_features import FrozenCLIP

def letterbox(image):
    image=image.convert('RGB');w,h=image.size
    if min(w,h)<1:raise ValueError('empty image')
    scale=224/max(w,h);size=(max(1,min(224,round(w*scale))),max(1,min(224,round(h*scale))))
    resized=image.resize(size,Image.Resampling.BICUBIC)
    result=Image.new('RGB',(224,224),(123,117,104))
    result.paste(resized,((224-size[0])//2,(224-size[1])//2));return result

class PairedCLIP(FrozenCLIP):
    def encode_images(self,images):
        self.verify_mode();out=[]
        with self.torch.inference_mode():
            for start in range(0,len(images),self.batch_size):
                batch=self.torch.stack([self.preprocess(image.convert('RGB')) for image in images[start:start+self.batch_size]])
                out.append(self.model.encode_image(batch.to(self.device)).float().cpu().numpy())
        value=np.concatenate(out).astype(np.float64);norm=np.linalg.norm(value,axis=1,keepdims=True)
        if not np.isfinite(value).all() or np.any(norm<=1e-6):raise ValueError('invalid CLIP image embedding')
        return (value/norm).astype(np.float32)
