#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from insightface.app import FaceAnalysis
from torchvision import transforms

# —————————————————————————————————————————
# CONFIG
# —————————————————————————————————————————
DATASET_DIR    = "/home/jetsonuser/masking/datasets/carla_v1/view_0"
IMG_NAME       = "pointcloud-0.png"
DEPTH_NAME     = "depth-0.png"
CONF_THRESH    = 0.5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE    = 10
DEPTH_STD_MULT = 75
ANON_BLOCK     = 16        # pixelation block size
ANON_NOISE     = 20        # noise level
OUT_RGB_PATH   = "rgb_anonymized.png"
OUT_DEPTH_PATH = "depth_anonymized.png"

# —————————————————————————————————————————
# HELPERS
# —————————————————————————————————————————
def detect_faces(img_bgr, app, conf_thresh):
    raw = app.get(img_bgr)
    faces = []
    for f in raw:
        if f.det_score < conf_thresh: continue
        x1, y1, x2, y2 = map(int, f.bbox)
        faces.append([x1, y1, x2, y2])
    return faces

def calc_depth_profile(depth_np, box, window=WINDOW_SIZE):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)//2, (y1+y2)//2
    hw = window//2
    win = depth_np[max(cy-hw,0):min(cy+hw,depth_np.shape[0]),
                   max(cx-hw,0):min(cx+hw,depth_np.shape[1])]
    vals = win.flatten()
    vals = vals[~np.isnan(vals)]
    if vals.size==0:
        return None
    m, s = float(vals.mean()), float(vals.std())
    return {'mean':m, 'th':s*DEPTH_STD_MULT, 'box':box}

def segment_mask(depth_t, profile):
    # depth_t: [H,W] tensor on DEVICE
    m = torch.tensor(profile['mean'], device=DEVICE)
    th= torch.tensor(profile['th'],   device=DEVICE)
    x1,y1,x2,y2 = profile['box']
    diff = torch.abs(depth_t - m)
    mask = (diff <= th).to(torch.uint8)
    # crop to box
    h,w = depth_t.shape
    x1_,x2_ = max(0,min(x1,w)),max(0,min(x2,w))
    y1_,y2_ = max(0,min(y1,h)),max(0,min(y2,h))
    out = torch.zeros_like(mask)
    out[y1_:y2_, x1_:x2_] = mask[y1_:y2_, x1_:x2_]
    return out

def anonymize_rgb(rgb_bgr, mask_t, block, noise):
    # rgb_bgr: H×W×3 uint8 BGR → tensor [3,H,W]
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    t = transforms.ToTensor()(rgb).to(DEVICE)  # [3,H,W], float in [0,1]
    # pixelate + noise
    C,H,W = t.shape
    p = F.avg_pool2d(t.unsqueeze(0), kernel_size=block, stride=block, ceil_mode=True)
    p = F.interpolate(p, size=(H,W), mode='nearest').squeeze(0)
    noise_t = (torch.rand_like(p)*2 - 1)*(noise/255.0)
    p = (p + noise_t).clamp(0,1)
    # blend
    m = mask_t.to(dtype=p.dtype).unsqueeze(0)
    out = p*m + t*(1-m)
    out = (out*255).byte().cpu().permute(1,2,0).numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def anonymize_depth_np(depth_np, mask_np, noise_strength):
    d = depth_np.copy()
    noise = np.random.normal(0, noise_strength, size=d.shape)
    d[mask_np.astype(bool)] += noise[mask_np.astype(bool)]
    return d

# —————————————————————————————————————————
# MAIN
# —————————————————————————————————————————
if __name__ == "__main__":
    # 1) Load images
    img_bgr   = cv2.imread( os.path.join(DATASET_DIR, IMG_NAME) )
    depth_np  = cv2.imread( os.path.join(DATASET_DIR, DEPTH_NAME),
                            cv2.IMREAD_UNCHANGED ).astype(np.float32)
    depth_t   = torch.from_numpy(depth_np).to(DEVICE)

    # 2) Init FaceAnalysis once
    app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection'])
    app.prepare(ctx_id=0 if DEVICE=='cuda' else -1, det_size=(2048,2048))

    # 3) Detect
    faces = detect_faces(img_bgr, app, CONF_THRESH)
    if not faces:
        print("No faces detected.")
        exit(0)

    # 4) Build combined mask
    combined = torch.zeros_like(depth_t, dtype=torch.uint8)
    for box in faces:
        prof = calc_depth_profile(depth_np, box)
        if prof is None: continue
        m = segment_mask(depth_t, prof)
        combined = torch.logical_or(combined.bool(), m.bool()).to(torch.uint8)

    mask_np = combined.cpu().numpy()

    # 5) Anonymize & save
    anon_rgb   = anonymize_rgb(img_bgr, combined, ANON_BLOCK, ANON_NOISE)
    anon_depth = anonymize_depth_np(depth_np, mask_np, noise_strength=10)

    cv2.imwrite(OUT_RGB_PATH,   anon_rgb)
    # save depth as 16-bit PNG
    cv2.imwrite(OUT_DEPTH_PATH, np.clip(anon_depth,0,65535).astype(np.uint16))

    print(f"Saved anonymized RGB → {OUT_RGB_PATH}")
    print(f"Saved anonymized depth→ {OUT_DEPTH_PATH}")