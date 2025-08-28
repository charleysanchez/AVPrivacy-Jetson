#!/usr/bin/env python3
import argparse
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response

cv2.setUseOptimized(True)
# cv2.setNumThreads(2)

# -------------------- CLI --------------------
p = argparse.ArgumentParser(description="RealSense live view → MJPEG over HTTP (Masked)")
p.add_argument("--save", action="store_true", help="Save frames to disk")
p.add_argument("--det-size", type=int, default=640, help="Detector input size (square)")
p.add_argument("--depth-anon", action="store_true", help="Also anonymize depth (off by default)")
p.add_argument("--port", type=int, default=5001, help="HTTP port for the MJPEG server")
p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for the stream (60–90 good range)")
p.add_argument("--preview-width", type=int, default=0, help="Resize stream to this width (0 = native)")
args = p.parse_args()

# -------------------- Import your masker --------------------
from av_privacy_masker import AVPrivacyMasker

mp = AVPrivacyMasker(
    device="cuda",
    conf_thresh=0.5,
    anon_block=24,
    anon_noise=20,
    dilate_kernel=13,
    det_size=(args.det_size, args.det_size),
    verbose=False,
    enable_depth_anon=args.depth_anon,
)

# -------------------- Folders --------------------
base_folder = 'images'
width = 640
height = 480
capture_fps = 30
output_fps = 30
hud = False

os.makedirs(base_folder, exist_ok=True)
session_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_time += f"_det-size{str(args.det_size)}"
output_folder = os.path.join(base_folder, f"session_{session_time}")
orig_dir = os.path.join(output_folder, "original")
anon_dir = os.path.join(output_folder, "anon")
if args.save:
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(anon_dir, exist_ok=True)

print(("Saving to" if args.save else "Use --save to save images to") + f": {output_folder}")

# -------------------- Camera (same pipeline as yours) --------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, capture_fps)
config.enable_stream(rs.stream.depth, width, height, rs.format.z16, capture_fps)

try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print(f"Error starting camera: {e}")
    sys.exit(1)

align = rs.align(rs.stream.color)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# -------------------- Warm-up detector (build TRT engine once) --------------------
dummy = np.zeros((height, width, 3), np.uint8)
_ = mp.detect_faces(dummy)

# -------------------- Capture thread --------------------
q = deque(maxlen=1)
stop_flag = False

def grabber():
    while not stop_flag:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        if not c or not d:
            continue
        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data())  # uint16 is fine for masking
        q.append((color, depth))

t_cap = threading.Thread(target=grabber, daemon=True)
t_cap.start()

# -------------------- HUD + processing state --------------------
frame_counter = 0
frames_to_skip = max(1, int(capture_fps / output_fps))
last_anon = None
t0 = time.perf_counter()
fps_smooth = None

def put_hud(img, text, y=22):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

# -------------------- Flask (MJPEG) --------------------
flask_app = Flask(__name__)
_latest_jpg = None
_latest_seq = 0
_jpg_lock = threading.Lock()
_new_frame_cv = threading.Condition(_jpg_lock)

@flask_app.route("/")
def index():
    return f"""
<html>
    <img src="/view" style="height:100%; width:auto; border-radius:8px;"/>
</html>
"""

@flask_app.route("/view")
def view_stream():
    def gen():
        boundary = b"--frame\r\n"
        headers = b"Content-Type: image/jpeg\r\n\r\n"
        last_sent = -1
        while True:
            with _jpg_lock:
                # wait until there is a newer frame than the last one we sent
                _new_frame_cv.wait_for(lambda: _latest_seq != last_sent)
                jpg_np = _latest_jpg
                last_sent = _latest_seq
            yield boundary + headers + jpg_np.tobytes() + b"\r\n"
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def publisher_loop():
    """Processes frames, builds side-by-side view, JPEG-encodes, publishes to Flask."""
    global frame_counter, last_anon, t0, fps_smooth, _latest_jpg, _latest_seq
    while not stop_flag:
        if not q:
            time.sleep(0.001)
            continue

        color, depth_raw = q[-1]

        detect_ms = mask_ms = anon_ms = 0

        tA = time.perf_counter()
        boxes = mp.detect_faces(color)
        detect_ms = int((time.perf_counter() - tA) * 1000)

        tB = time.perf_counter()
        mask_np = mp.build_mask_numpy(depth_raw, boxes, mp.kernel, mp._calc_depth_profile)
        mask_ms = int((time.perf_counter() - tB) * 1000)

        tC = time.perf_counter()
        rgb_anon = mp.fast_pixelate(color, mask_np, block=mp.anon_block, noise=mp.anon_noise)
        anon_ms = int((time.perf_counter() - tC) * 1000)

        last_anon = rgb_anon

        # Compose side-by-side
        # view = np.hstack([color, rgb_anon])
        view = rgb_anon


        # Optional resize for lighter streaming
        if args.preview_width and args.preview_width > 0:
            h = int(view.shape[0] * (args.preview_width / view.shape[1]))
            view = cv2.resize(view, (args.preview_width, h), interpolation=cv2.INTER_AREA)

        # FPS HUD
        dt = time.perf_counter() - t0
        t0 = time.perf_counter()
        curr_fps = 1.0 / max(1e-6, dt)
        fps_smooth = curr_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * curr_fps)
        if hud:
            put_hud(view, f"FPS ~ {fps_smooth:.1f}")
            put_hud(view, f"det {detect_ms}ms | mask {mask_ms}ms | anon {anon_ms}ms", y=44)

        # Save every Nth frame (optional)
        if args.save and (frame_counter % frames_to_skip == 0):
            ts = int(time.time() * 1000)
            cv2.imwrite(os.path.join(orig_dir, f"frame_{ts}.jpg"), color)
            cv2.imwrite(os.path.join(anon_dir, f"frame_{ts}.jpg"), rgb_anon)

        # Encode JPEG for streaming
        ok, buf = cv2.imencode(".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if ok:
            with _jpg_lock:
                _latest_jpg = buf  # keep as numpy array; convert to bytes at yield
                _latest_seq += 1
                _new_frame_cv.notify_all()

        frame_counter += 1

# Start publisher thread
t_pub = threading.Thread(target=publisher_loop, daemon=True)
t_pub.start()

# -------------------- Run Flask server --------------------
try:
    print(f"Open http://localhost:{args.port}/  (or ssh tunnel: ssh -L {args.port}:localhost:{args.port} <user>@<jetson>)")
    flask_app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
except KeyboardInterrupt:
    pass
finally:
    stop_flag = True
    time.sleep(0.1)
    pipeline.stop()