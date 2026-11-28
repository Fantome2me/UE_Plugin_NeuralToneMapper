"""Relative HDR RGB [0,1] -> SDR sRGB. No absolute-nit HDR output is implied."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def up(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class LocalLUT(nn.Module):
    def __init__(self, n=17):
        super().__init__()
        b, g, r = torch.meshgrid(*[torch.linspace(0, 1, n)] * 3, indexing='ij')
        identity = torch.stack([r, g, b]).clamp_min(0).pow(1/2.2)
        self.tables = nn.Parameter(identity.expand(3, -1, -1, -1, -1).clone())
        self.embed = nn.Conv2d(3, 16, 3, padding=1)
        self.attend = nn.MultiheadAttention(16, 2, batch_first=True)
        self.weights = nn.Conv2d(16, 3, 1)
        self.detail = nn.Sequential(nn.Conv2d(7, 12, 3, padding=1), nn.ReLU(), nn.Conv2d(12, 2, 1))

    def forward(self, x):
        levels = []
        low = x
        while max(low.shape[-2:]) > 64:
            nxt = F.avg_pool2d(low, 2, ceil_mode=True, count_include_pad=False)
            levels.append((low, low - up(nxt, low.shape[-2:])))
            low = nxt
        feat = self.embed(torch.sqrt(low.clamp_min(1e-8)))
        tokens = F.adaptive_avg_pool2d(feat, (8, 8)).flatten(2).transpose(1, 2)
        q = feat.flatten(2).transpose(1, 2)
        a = self.attend(q, tokens, tokens, need_weights=False)[0].transpose(1, 2).reshape_as(feat)
        w = self.weights(feat + a).softmax(1)
        grid = (low.permute(0, 2, 3, 1) * 2 - 1).unsqueeze(1)
        y = sum(w[:,i:i+1] * F.grid_sample(t.expand(x.size(0),-1,-1,-1,-1), grid,
                align_corners=True, padding_mode='border').squeeze(2) for i,t in enumerate(self.tables))
        for src, band in reversed(levels):
            y = up(y, src.shape[-2:])
            edge = band.abs().mean(1, keepdim=True)
            gain, sigma = self.detail(torch.cat([src, y, edge], 1)).chunk(2, 1)
            sigma = 0.002 + 0.2 * sigma.sigmoid()
            y = y + band * (1 + 0.9 * gain.tanh() * torch.exp(-edge/sigma))
        return y


def load(root, ids, size):
    xx, yy = [], []
    for i in ids:
        x = cv2.imread(str(root/'middle_480p'/(i+'.png')), -1)
        y = cv2.imread(str(root/'output_480p'/(i+'.jpg')))
        assert x.dtype == np.uint16 and x.shape == y.shape
        for a, dest, scale in [(x, xx, 65535), (y, yy, 255)]:
            a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
            dest.append(torch.from_numpy(a[:,:,::-1].copy()).permute(2,0,1).float()/scale)
    return torch.stack(xx), torch.stack(yy)


def run(root, out, epochs):
    torch.manual_seed(114)
    ids = lambda name: (root/(name+'.txt')).read_text().split()
    train, val = ids('train'), ids('valid')
    test = sorted(set(ids('test'))-set(val))
    assert not set(train) & (set(val)|set(test))
    shape=(160,96)
    x,y = load(root, train, shape)
    vx,vy = [a.cuda() for a in load(root, val, shape)]
    m = LocalLUT().cuda(); opt = torch.optim.Adam(m.parameters(), lr=0.002)
    best = float('inf'); ck = out/'llf_sdr_candidate.pt'
    for epoch in range(epochs):
        m.train()
        for idx in torch.randperm(len(train)).split(8):
            src, ref = x[idx].cuda(), y[idx].cuda()
            p = m(src)
            tv = sum(m.tables.diff(dim=d).square().mean() for d in (2,3,4))
            loss = (p-ref).abs().mean()+0.12*(p.diff(dim=-1)-ref.diff(dim=-1)).abs().mean()+0.01*tv
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            m.eval(); err = (m(vx).clamp(0,1)-vy).abs().mean().item()
            if err < best:
                best = err;torch.save(m.state_dict(),ck)
        print(epoch+1,err,flush=True)
    m.load_state_dict(torch.load(ck,weights_only=True)); m.eval()
    errs=[]; base=[]
    with torch.no_grad():
        for j in range(0,len(test),8):
            u,v=[a.cuda() for a in load(root,test[j:j+8],shape)]
            errs.extend((m(u).clamp(0,1)-v).abs().mean((1,2,3)).tolist())
            base.extend((u.pow(1/2.2)-v).abs().mean((1,2,3)).tolist())
    report=dict(gpu=torch.cuda.get_device_name(),train=len(train),valid=len(val),test=len(test),size=shape,
                epochs=epochs,val_mae=best,test_mae=np.mean(errs),gamma_only_mae=np.mean(base))
    (out/'llf_sdr_metrics.json').write_text(json.dumps(report,indent=2));print(report)


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--epochs',type=int,default=8)
    args=ap.parse_args();run(args.root,Path(__file__).resolve().parent,args.epochs)
