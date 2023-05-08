import torch
import s3prl.hub as hub

ckpt = "/data/part4/checkpoint_best1.pt"
extracter = getattr(hub, 'hubert_local')(ckpt)

wavs = [torch.zeros(160000, dtype=torch.float) for _ in range(16)]
with torch.no_grad():
    mfcc = extracter(wavs)["hidden_states"] 