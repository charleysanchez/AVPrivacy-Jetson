#!/usr/bin/env python3
import onnxruntime as ort
ort.set_default_logger_severity(4)   # silence ONNX-Runtime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyrealsense2 as rs
from flask import Flask, Response
from insightface.app import FaceAnalysis
from torchvision import transforms

# —————————————————————————————————————————
# CONFIG
# —————————————————————————————————————————
CONF_THRESH    = 0.5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DET_SIZE       = (640, 640)
WINDOW_SIZE    = 10
DEPTH_STD_MULT = 75
ANON_BLOCK     = 16
ANON_NOISE     = 20
FLASK_PORT     = 5000

# —————————————————————————————————————————
# FACE DETECTOR (load once)
# —————————————————————————————————————————
app_insight = FaceAnalysis(name='buffalo_s', allowed_modules=['detection'])
app_insight.prepare(ctx_id=0 if DEVICE=='cuda' else -1, det_size=DET_SIZE)

# —————————————————————————————————————————
# STREAM GRABBER (will be bound in __main__)
# —————————————————————————————————————————
def grab_frames():
    # placeholder; real one bound in __main__
    yield from ()

# —————————————————————————————————————————
# ANON HELPERS
# —————————————————————————————————————————
def detect_faces(img):
    small = cv2.resize(img, DET_SIZE)
    raw   = app_insight.get(small)
    h0,w0 = img.shape[:2]
    fx,fy = w0/DET_SIZE[0], h0/DET_SIZE[1]
    boxes = []
    for f in raw:
        if f.det_score < CONF_THRESH: continue
        x1,y1,x2,y2 = map(int, f.bbox)
        boxes.append([int(x1*fx), int(y1*fy), int(x2*fx), int(y2*fy)])
    return boxes

def build_mask(depth_np, depth_t, boxes):
    mask = torch.zeros_like(depth_t, dtype=torch.uint8)
    for x1,y1,x2,y2 in boxes:
        cx,cy = (x1+x2)//2, (y1+y2)//2
        win = depth_np[
            max(cy-WINDOW_SIZE//2,0):min(cy+WINDOW_SIZE//2,depth_np.shape[0]),
            max(cx-WINDOW_SIZE//2,0):min(cx+WINDOW_SIZE//2,depth_np.shape[1])
        ].flatten()
        win = win[~np.isnan(win)]
        if win.size==0: continue
        m   = float(win.mean())
        th  = float(win.std()*DEPTH_STD_MULT)
        diff= torch.abs(depth_t - m)
        region = (diff<=th).to(torch.uint8)
        H,W = depth_t.shape
        x1_,x2_ = max(0,min(x1,W)), max(0,min(x2,W))
        y1_,y2_ = max(0,min(y1,H)), max(0,min(y2,H))
        mask[y1_:y2_, x1_:x2_] |= region[y1_:y2_, x1_:x2_]
    m_np = mask.cpu().numpy().astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    return cv2.dilate(m_np, kern)

def anonymize_rgb(img, mask_np):
    mask_t = torch.from_numpy(mask_np).to(DEVICE)
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t      = transforms.ToTensor()(rgb).to(DEVICE)
    p = F.avg_pool2d(t.unsqueeze(0), ANON_BLOCK, ANON_BLOCK, ceil_mode=True)
    p = F.interpolate(p, size=t.shape[1:], mode='nearest').squeeze(0)
    noise = (torch.rand_like(p)*2 - 1)*(ANON_NOISE/255.0)
    p = (p+noise).clamp(0,1)
    m = mask_t.to(dtype=p.dtype).unsqueeze(0)
    out = p*m + t*(1-m)
    out = (out*255).byte().cpu().permute(1,2,0).numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

def anonymize_depth(depth_np, mask_np):
    d     = depth_np.copy()
    noise = np.random.normal(0, ANON_NOISE, size=d.shape)
    d[mask_np==1] += noise[mask_np==1]
    vis   = cv2.normalize(d, None, 0,255, cv2.NORM_MINMAX)
    return vis.astype(np.uint8)

# —————————————————————————————————————————
# FLASK APP
# —————————————————————————————————————————
flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return """
    <html>
      <head>
        <title>4-Stream Anon</title>
        <style>
          img { width: 40cd%; height: auto; display: inline-block; }
        </style>
      </head>
      <body>
        <h1>Real-Time Streams</h1>
        <p>
          <img src="/rgb_feed"      alt="Raw RGB" />
          <img src="/depth_feed"    alt="Raw Depth" />
        </p>
        <p>
          <img src="/rgb_anon_feed"  alt="Anon RGB" />
          <img src="/depth_anon_feed"alt="Anon Depth" />
        </p>
      </body>
    </html>
    """

def make_mjpeg(gen_func):
    return Response(gen_func(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@flask_app.route("/rgb_feed",      strict_slashes=False)
def rgb_feed():
    def gen():
        for color, _ in grab_frames():
            _, jpg = cv2.imencode('.jpg', color)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpg.tobytes() + b'\r\n')
    return make_mjpeg(gen)

@flask_app.route("/depth_feed",    strict_slashes=False)
def depth_feed():
    def gen():
        for _, depth in grab_frames():
            vis = cv2.normalize(depth, None, 0,255, cv2.NORM_MINMAX)
            _, jpg = cv2.imencode('.jpg', vis.astype(np.uint8))
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpg.tobytes() + b'\r\n')
    return make_mjpeg(gen)

@flask_app.route("/rgb_anon_feed", strict_slashes=False)
def rgb_anon_feed():
    def gen():
        for color, depth in grab_frames():
            depth_t = torch.from_numpy(depth).to(DEVICE)
            mask    = build_mask(depth, depth_t, detect_faces(color))
            anon    = anonymize_rgb(color, mask)
            _, jpg = cv2.imencode('.jpg', anon)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpg.tobytes() + b'\r\n')
    return make_mjpeg(gen)

@flask_app.route("/depth_anon_feed", strict_slashes=False)
def depth_anon_feed():
    def gen():
        for _, depth in grab_frames():
            depth_t = torch.from_numpy(depth).to(DEVICE)
            mask    = build_mask(depth, depth_t, detect_faces(color=None) 
                                 if False else build_mask)  # dummy
            # Actually we need color for detect_faces; so better pass color
            # Let's correct: we need both color & depth.
            break
    # Oops: logic needs both color & depth, so just iterate both:
    def gen():
        for color, depth in grab_frames():
            depth_t = torch.from_numpy(depth).to(DEVICE)
            mask    = build_mask(depth, depth_t, detect_faces(color))
            anon    = anonymize_depth(depth, mask)
            _, jpg = cv2.imencode('.jpg', anon)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpg.tobytes() + b'\r\n')
    return make_mjpeg(gen)

if __name__ == "__main__":
    # initialize RealSense
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 30)
    profile  = pipeline.start(cfg)
    align    = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    def grab_frames():
        while True:
            frames = align.process(pipeline.wait_for_frames())
            c = frames.get_color_frame()
            d = frames.get_depth_frame()
            if not c or not d:
                continue
            img   = np.asanyarray(c.get_data())
            depth = np.asanyarray(d.get_data()).astype(np.float32)*depth_scale
            yield img, depth

    try:
        flask_app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True)
    finally:
        pipeline.stop()