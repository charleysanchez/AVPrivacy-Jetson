#!/usr/bin/env python3

import os
import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
from insightface.app import FaceAnalysis
from torchvision import transforms

# —————————————————————————————————————————
# CONFIG
# —————————————————————————————————————————
DATASET_DIR   = "/home/jetsonuser/masking/datasets/carla_v1/view_0"
IMG_NAME      = "pointcloud-0.png"
DEPTH_NAME    = "depth-0.png"
CONF_THRESH   = 0.5
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE   = 10
DEPTH_STD_MULT= 75
ANON_BLOCK    = 16
ANON_NOISE    = 20
OUT_RGB       = "rgb_anonymized.png"
OUT_DEPTH= "depth_anonymized.png"

# —————————————————————————————————————————
# HELPERS
# —————————————————————————————————————————
def detect_faces(img, app, thresh):
    raw = app.get(img)
    boxes = []
    for f in raw:
        if f.det_score < thresh: continue
        x1,y1,x2,y2 = map(int, f.bbox)
        boxes.append([x1,y1,x2,y2])
    return boxes

def calc_depth_profile(depth_np, box, w=WINDOW_SIZE):
    x1,y1,x2,y2 = box
    cx,cy = (x1+x2)//2, (y1+y2)//2
    hw = w//2
    win = depth_np[
        max(cy-hw,0):min(cy+hw,depth_np.shape[0]),
        max(cx-hw,0):min(cx+hw,depth_np.shape[1])
    ].flatten()
    win = win[~np.isnan(win)]
    if win.size==0: 
        return None
    m,s = float(win.mean()), float(win.std())
    return {'mean':m, 'th':s*DEPTH_STD_MULT, 'box':box}

def segment_mask(depth_t, prof):
    m = torch.tensor(prof['mean'],device=DEVICE)
    th= torch.tensor(prof['th'],   device=DEVICE)
    diff = torch.abs(depth_t - m)
    mask = (diff <= th).to(torch.uint8)
    x1,y1,x2,y2 = prof['box']
    H,W = depth_t.shape
    # clamp & crop
    x1_,x2_ = max(0,min(x1,W)), max(0,min(x2,W))
    y1_,y2_ = max(0,min(y1,H)), max(0,min(y2,H))
    out = torch.zeros_like(mask)
    out[y1_:y2_, x1_:x2_] = mask[y1_:y2_, x1_:x2_]
    return out

def anonymize_rgb(img_bgr, mask_t):
    # to tensor
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = transforms.ToTensor()(rgb).to(DEVICE)  # [3,H,W] float 0–1
    C,H,W = t.shape
    # pixelate+noise
    p = F.avg_pool2d(t.unsqueeze(0), ANON_BLOCK, ANON_BLOCK, ceil_mode=True)
    p = F.interpolate(p, size=(H,W), mode='nearest').squeeze(0)
    noise = (torch.rand_like(p)*2-1)*(ANON_NOISE/255.0)
    p = (p+noise).clamp(0,1)
    # blend
    m = mask_t.to(dtype=p.dtype).unsqueeze(0)
    out = p*m + t*(1-m)
    out = (out*255).byte().cpu().permute(1,2,0).numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def anonymize_depth(depth_np, mask_np, noise=10):
    d = depth_np.copy()
    # only add noise where mask==1
    d[mask_np.astype(bool)] += np.random.normal(0, noise, size=d.shape)[mask_np.astype(bool)]
    return d

# —————————————————————————————————————————
# MAIN
# —————————————————————————————————————————
if __name__=="__main__":
    # load
    img = cv2.imread(os.path.join(DATASET_DIR, IMG_NAME))
    depth = cv2.imread(os.path.join(DATASET_DIR, DEPTH_NAME),
                      cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth_t = torch.from_numpy(depth).to(DEVICE)

    # face detector
    app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection'])
    app.prepare(ctx_id=0 if DEVICE=='cuda' else -1, det_size=(2048, 2048))

    start_t = time.perf_counter()
    # detect & profile
    boxes = detect_faces(img, app, CONF_THRESH)
    print("Detected boxes:", boxes)
    combined = torch.zeros_like(depth_t, dtype=torch.uint8)

    for box in boxes:
        prof = calc_depth_profile(depth, box)
        if prof is None:
            print("→ empty depth window for box", box)
            continue
        m = segment_mask(depth_t, prof)
        combined = torch.logical_or(combined.bool(), m.bool()).to(torch.uint8)

    mask_np = combined.cpu().numpy()

    # debug & dilation
    print("Mask pixels:", mask_np.sum(), "/", mask_np.size)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    mask_dil = cv2.dilate(mask_np.astype(np.uint8), kern)

    # anonymize
    rgb_anon   = anonymize_rgb(img, torch.from_numpy(mask_dil).to(DEVICE))
    depth_anon = anonymize_depth(depth, mask_dil, noise=10)

    # save RGB
    cv2.imwrite(OUT_RGB, rgb_anon)

    # save raw 16-bit (for downstream) AND a normalized 8-bit for viewing
    norm = cv2.normalize(depth_anon, None, 0,255, cv2.NORM_MINMAX)
    cv2.imwrite(OUT_DEPTH, norm.astype(np.uint8))

    print(f"total time for pipeline (after model instantiation): {time.perf_counter() - start_t:.2f}")

    print("→ RGB anonymized →", OUT_RGB)
    print("→ depth viewable  →", OUT_DEPTH)
