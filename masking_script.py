#!/usr/bin/env python3

from insightface.app import FaceAnalysis
from PIL import Image
from torch.utils.data import Dataset

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

DATASET_DIR = "/home/jetsonuser/masking/datasets/carla_v1/view_0"
DETECTION_CONFIDENCE_THRESHOLD = 0.3
PRIVATE_OBJECT_CLASSES = ['person']  # used downstream by segment_all
WINDOW_SIZE = 10
DEPTH_THRESHOLD_MULTIPLIER = 75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 1
NUM_FRAMES = 1
NUM_VIEWS = 1

img_name = "pointcloud-0.png"

img_path = os.path.join(DATASET_DIR, img_name)
pointcloud_np = cv2.imread(img_path)

depth_path = os.path.join(DATASET_DIR, 'depth-0.png')
depth_np = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
depth_tensor = torch.from_numpy(depth_np).to("cuda")

# initialize once at top-level
retina_app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection'])
# ctx_id=0 for GPU, -1 for CPU
retina_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1,
                   det_size=(2048,2048))

def detect_faces(image_np, confidence_threshold=0.5, draw_boxes=False):
    """
    image_np: BGR uint8 H×W×3
    Returns list of {'box':[x1,y1,x2,y2],'score':…,'label':'person'}
    or, if draw_boxes=True, returns the BGR image with boxes overlaid.
    """
    # Step 1: Detect
    raw_faces = retina_app.get(image_np)
    faces = []
    for f in raw_faces:
        score = f.det_score
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(int, f.bbox)
        faces.append({
                "box": [x1, y1, x2, y2],
                "score": float(score),
                "label": "person"
            })
        
    if not draw_boxes:
        return faces
    
    # Step 2: Draw
    out = image_np.copy()
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        score = face["score"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score:.2f}", (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return out


def calculate_depth_profile_of_box(depth_map, x1, y1, x2, y2, window_size=WINDOW_SIZE):
    """
    Return { 'mean','std','threshold','box':[x1,y1,x2,y2] } or None if empty.
    """
    half_window = window_size // 2
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    x_start = max(cx - half_window, 0)
    y_start = max(cy - half_window, 0)
    x_end = min(cx + half_window, depth_map.shape[1])
    y_end = min(cy + half_window, depth_map.shape[0])

    depth_window = depth_map[y_start:y_end, x_start:x_end]
    depth_values = depth_window.flatten()
    depth_values = depth_values[~np.isnan(depth_values)]
    if depth_values.size == 0:
        return None
    
    depth_mean = float(np.mean(depth_values))
    depth_std = float(np.std(depth_values))
    depth_threshold = float(depth_std * DEPTH_THRESHOLD_MULTIPLIER)

    return {
        'mean': depth_mean,
        'std': depth_std,
        'threshold': depth_threshold,
        'box': [x1, y1, x2, y2]
    }

def segment_person_from_box(depth_tensor, depth_profile):
    """
    Similar to segment_person_from_profile_batch, but for a single bounding box
    in just 1 frame's depth or multiple frames (X frames).
    If depth_tensor: shape [X,H,W] or [H,W].
    """
    if len(depth_tensor.shape) == 2:
        # single frame [H, W]
        depth_tensor = depth_tensor.unsqueeze(0) # [1, H, W]
    
    depth_mean = torch.tensor(depth_profile['mean'], device=depth_tensor.device, dtype=depth_tensor.dtype)
    depth_threshold = torch.tensor(depth_profile['threshold'], device=depth_tensor.device, dtype=depth_tensor.dtype)
    (x1, y1, x2, y2) = depth_profile['box']
    y2 = int(y1 * (y2-y1))

    depth_diff = torch.abs(depth_tensor - depth_mean)
    mask_batch = (depth_diff <= depth_threshold).to(torch.uint8)

    final_mask = torch.zeros_like(mask_batch)
    _, H, W = depth_tensor.shape
    x1_clamp = max(0, min(x1, W))
    x2_clamp = max(0, min(x2, W))
    y1_clamp = max(0, min(y1, H))
    y2_clamp = max(0, min(y2, H))

    if x2_clamp > x1_clamp and y2_clamp > y1_clamp:
        final_mask[:, y1_clamp:y2_clamp, x1_clamp:x2_clamp] = mask_batch[:, y1_clamp:y2_clamp, x1_clamp:x2_clamp]

    # return shape [H, W] if a single frame
    if final_mask.shape[0] == 1:
        return final_mask[0]
    return final_mask


def segment_all(depth_tensor, faces, depth_map):
    """
    We create a combined mask of shape [H,W] = 1 for each person's bounding box,
    EXCEPT we skip the public_box (which is the "public" person).
    depth_tensor: shape [H,W], float on GPU
    objects: detection results on CPU
    public_box: (x1,y1,x2,y2) that we skip
    depth_map: CPU 2D array for depth
    Return: torch.uint8 mask [H,W], 1=private, 0=public
    """
    print("depth_tensor dtype:", depth_tensor.dtype, "min:", depth_tensor.min().item(), "max:", depth_tensor.max().item())
    H, W = depth_tensor.shape[-2], depth_tensor.shape[-1]

    combined_mask = torch.zeros((H, W), dtype=torch.uint8, device=depth_tensor.device)

    for face in faces:
        if face['label'] not in PRIVATE_OBJECT_CLASSES:
            continue
        box = face['box']

        depth_profile = calculate_depth_profile_of_box(depth_map, *box)
        if depth_profile is None:
            print("depth_profile is None")
            continue

        single_mask = segment_person_from_box(depth_tensor, depth_profile)
        print("Single mask nonzero:", torch.any(single_mask != 0).item())
        combined_mask = torch.logical_or(combined_mask.bool(), single_mask.bool()).to(torch.uint8)

    return combined_mask


def dice_score_batch(pred_batch, gt_batch):
    intersection = torch.sum((pred_batch==1)&(gt_batch==1)).item()
    pred_sum = torch.sum(pred_batch==1).item()
    gt_sum = torch.sum(gt_batch==1).item()
    if (pred_sum+gt_sum)==0:
        return 1.0
    return 2.0*intersection/(pred_sum+gt_sum)

def recall_batch(pred_batch, gt_batch):
    tp = torch.sum((pred_batch==1)&(gt_batch==1)).item()
    gt_sum = torch.sum(gt_batch==1).item()
    if gt_sum==0:
        return 1.0
    return tp/gt_sum


def anonymize_region(img_tensor, mask_tensor, block, noise_level):
    """
    Optimized GPU anonymization: pixelates + noise on the masked region.
    Works without padding errors by extracting the ROI mask.
    """
    # 1) Prepare image in float [0,1]
    orig_dtype = img_tensor.dtype
    img = img_tensor.float().to(img_tensor.device)
    if orig_dtype == torch.uint8:
        img = img / 255.0

    # 2) Find ROI bounds from full-image mask
    ys, xs = mask_tensor.nonzero(as_tuple=True)
    if ys.numel() == 0:
        return img_tensor  # nothing to anonymize
    
    y1, y2 = ys.min().item(), ys.max().item() + 1
    x1, x2 = xs.min().item(), xs.max().item() + 1

    # 3) Crop ROI from image and mask
    roi = img[:, y1:y2, x1:x2]               # [C, h, w]
    mask_roi = mask_tensor[y1:y2, x1:x2]     # [h, w]
    C, h, w = roi.shape

    # 4) Pixelate via avg pool → nearest upsample (ceil_mode handles edges)
    pooled = F.avg_pool2d(
        roi.unsqueeze(0),               # [1, C, h, w]
        kernel_size=block,
        stride=block,
        ceil_mode=True
    )                                   # → [1, C, ceil(h/block), ceil(w/block)]
    mosaic = F.interpolate(
        pooled,
        size=(h, w),
        mode='nearest'
    ).squeeze(0)                        # → [C, h, w]

    # 5) Add uniform noise to mosaic blocks
    noise = (torch.rand_like(mosaic) * 2 - 1) * (noise_level / 255.0)
    mosaic_noised = (mosaic + noise).clamp(0.0, 1.0)

    # 6) Blend only the masked pixels in the ROI
    mask_f = mask_roi.to(dtype=mosaic_noised.dtype, device=img.device)      # [h, w]
    mask_exp = mask_f.unsqueeze(0).expand(C, h, w)                         # [C, h, w]
    region = mosaic_noised * mask_exp + roi * (1.0 - mask_exp)             # [C, h, w]

    # 7) Write region back into a copy of the full image
    out = img.clone()
    out[:, y1:y2, x1:x2] = region

    # 8) Convert back to uint8 if needed
    if orig_dtype == torch.uint8:
        out = (out * 255.0).round().to(torch.uint8)

    return out


def anonymize_depth(depth_np, mask, noise_strength):
    """
    Add noise to depth data to anonymize the region. Only applies to mask area if mask is given.
    depth_np: H×W depth array (float32).
    mask: H×W boolean array for region to anonymize (same size as depth_np).
    """
    depth_out = depth_np.copy()
    if mask is None:
        # If no mask provided, apply to whole depth (not recommended for performance)
        mask = np.ones_like(depth_out, dtype=bool)
    # You can add just one type of noise for speed. Here we use Gaussian.
    noise = np.random.normal(loc=0.0, scale=noise_strength, size=depth_out.shape)
    depth_out[mask] += noise[mask]
    # Optionally clamp or otherwise limit values if needed (e.g., keep depth in plausible range).
    return depth_out


def save_masked_images(pred_mask_full, images, out_folder, dilation_radius=4):
    """
    Saves masked RGB & depth frames to disk and records timing metrics.

    Args:
        pred_mask_full (torch.Tensor): uint8 mask tensor of shape [V, F, H, W]
        images (torch.Tensor): image tensor of shape [V, F, 4, H, W] (RGB + depth)
        out_folder (str): base directory where `rgb/view*/` and `depth/view*/` will be created
        dilation_radius (int): radius (in px) for dilating the mask before anonymization

    Returns:
        dict: {
            'chunk_total': [float],  # total time per chunk
            'chunk_anon': [float],  # anonymization time per chunk
            'chunk_write': [float]  # file-write time per chunk
        }
    """
    # 1) Prepare output dirs for each view
    os.makedirs(out_folder, exist_ok=True)
    V, F, H, W = pred_mask_full.shape
    for v in range(V):
        os.makedirs(os.path.join(out_folder, "rgb", f"view{v}"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "depth", f"view{v}"), exist_ok=True)

    # 2) Precompute the CPU-side dilation kernel
    k = 2 * dilation_radius + 1
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # 3) Containers for timing
    chunk_total_times = []
    chunk_anon_times = []
    chunk_write_times = []

    # 4) Process in chunks of CHUNK_SIZE frames
    for start in range(0, F, CHUNK_SIZE):
        # Sync GPU before starting chunk timer
        torch.cuda.synchronize()
        t_chunk = time.perf_counter()

        anon_times = []
        write_times = []
        end = min(start + CHUNK_SIZE, F)

        # Loop over each view and frame
        for v in range(V):
            for f in range(start, end):
                # --- A) Dilate mask on CPU ---
                mask_np = pred_mask_full[v, f].cpu().numpy().astype(np.uint8)
                mask_dilated = cv2.dilate(mask_np, dilation_kernel).astype(bool)

                # --- B) Grab depth frame for later save ---
                depth_frame = images[v, f, 3].cpu().numpy()

                # --- C) Prepare GPU tensors ---
                rgb_gpu = images[v, f, :3]                         # [3,H,W]
                mask_gpu = torch.from_numpy(mask_dilated).to(rgb_gpu.device)  # [H,W]

                # --- D) Anonymize ONCE per chunk (at f == start) ---
                if f == start:
                    torch.cuda.synchronize()
                    t_anon = time.perf_counter()

                    anonymized_gpu = anonymize_region(
                        rgb_gpu, mask_gpu,
                        block=max(1, W // 16),
                        noise_level=20
                    )

                    # Bring result to CPU for compositing
                    anon_arr = (anonymized_gpu
                                .permute(1, 2, 0)
                                .cpu()
                                .numpy())
                    if anon_arr.dtype != np.uint8:
                        anon_arr = (anon_arr * 255).clip(0, 255).astype(np.uint8)

                    anon_times.append(time.perf_counter() - t_anon)

                # --- E) Composite anonymized region into original RGB ---
                orig = (rgb_gpu
                        .permute(1, 2, 0)
                        .cpu()
                        .numpy())
                if orig.dtype != np.uint8:
                    orig = (orig * 255).clip(0, 255).astype(np.uint8)
                out_rgb = orig.copy()
                out_rgb[mask_dilated] = anon_arr[mask_dilated]

                # --- F) Write images & record write time ---
                t_write = time.perf_counter()
                # RGB
                cv2.imwrite(
                    os.path.join(out_folder, "rgb", f"view{v}", f"{f}_masked.png"),
                    cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                )
                # Depth
                depth_u16 = np.clip(
                    depth_frame + np.random.normal(0, 10, depth_frame.shape),
                    0, 20000
                ).astype(np.uint16)
                cv2.imwrite(
                    os.path.join(out_folder, "depth", f"view{v}", f"{f}_depth.png"),
                    depth_u16
                )
                write_times.append(time.perf_counter() - t_write)

        # Sync GPU and record chunk timings
        torch.cuda.synchronize()
        chunk_total_times.append(time.perf_counter() - t_chunk)
        chunk_anon_times.append(sum(anon_times))
        chunk_write_times.append(sum(write_times))

    return {
        'chunk_total': chunk_total_times,
        'chunk_anon': chunk_anon_times,
        'chunk_write': chunk_write_times
    }

if __name__ == '__main__':
    faces = detect_faces(pointcloud_np, confidence_threshold=0.5, draw_boxes=False)
    print(faces)

    f_box = detect_faces(pointcloud_np, confidence_threshold=0.5, draw_boxes=True)
    cv2.imwrite("test.png", f_box)


    mask = segment_all(depth_tensor, faces, depth_np)
    print(mask, mask.shape, bool(torch.any(mask != 0)), mask.max().item())
    mask_array = mask.mul(255).byte().cpu().numpy()
    img = Image.fromarray(mask_array)
    img.save("test_segment.png")

    # Assign H, W
    H, W = pointcloud_np.shape[-2], pointcloud_np.shape[-1]

    # Allocate mask
    pred_mask_full = torch.zeros((NUM_VIEWS, NUM_FRAMES, H, W), dtype=torch.uint8, device=DEVICE)

    # Detection + Segmentation
    chunk_detection_times, chunk_seg_times, chunk_total_times = [], [], []
    chunk_starts = list(range(0, NUM_FRAMES, CHUNK_SIZE))
    print(f"Chunk starts: {chunk_starts}")
    for start_f in chunk_starts:
        end_f = min(start_f+CHUNK_SIZE, NUM_FRAMES)
        t_chunk = time.perf_counter()
        for v in range(NUM_VIEWS):
            # rgb_t = images[v,start_f,:3]
            # rgb_np = (rgb_t.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            # bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            pointcloud_np

            t_d = time.perf_counter()
            dets = detect_faces(pointcloud_np, DETECTION_CONFIDENCE_THRESHOLD)
            boxes = [[*det['box']] for det in dets]
            chunk_detection_times.append(time.perf_counter()-t_d)

            t_s = time.perf_counter()
            private_mask = segment_all(depth_tensor, [{'box':b,'label':'person'} for b in boxes], depth_np)
            chunk_seg_times.append(time.perf_counter()-t_s)
            pred_mask_full[v,start_f:end_f] = private_mask
        chunk_total_times.append(time.perf_counter()-t_chunk)

    # Print timings
    print("Detection times per chunk:", chunk_detection_times)
    print("Segmentation times per chunk:", chunk_seg_times)
    print("Total times per chunk:", chunk_total_times)

    print(f"Avg detection time     = {sum(chunk_detection_times)/len(chunk_detection_times):.4f}s")
    print(f"Avg segmentation time  = {sum(chunk_seg_times)/len(chunk_starts):.4f}s")
    print(f"Avg total time/chunk   = {sum(chunk_total_times)/len(chunk_starts):.4f}s")

    # Save anonymized images + timings
    timings = save_masked_images_gpu(pred_mask_full, images, output_base_directory, dilation_radius=4)
    print("Per-chunk totals:", timings['chunk_total'])
    print("Per-chunk anonym times:", timings['chunk_anon'])
    print("Per-chunk write times:", timings['chunk_write'])
    print("Avg total:", np.mean(timings['chunk_total']))
    print("Avg anon:",  np.mean(timings['chunk_anon']))
    print("Avg write:", np.mean(timings['chunk_write'])
