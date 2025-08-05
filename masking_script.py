from insightface.app import FaceAnalysis
from PIL import Image
from torch.utils.data import Dataset

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transforms

DATASET_DIR = "/home/jetsonuser/masking/datasets/carla_v1/view_0"
DETECTION_CONFIDENCE_THRESHOLD = 0.3
PRIVATE_OBJECT_CLASSES = ['person']  # used downstream by segment_all
WINDOW_SIZE = 10
DEPTH_THRESHOLD_MULTIPLIER = 75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def segment_person_from_box(depth_tensor, depth_profile, span=1):
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
    y2 = int(y1 + span * (y2-y1))

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


def segment_all(depth_tensor, faces, depth_map, span):
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

        single_mask = segment_person_from_box(depth_tensor, depth_profile, span)
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


if __name__ == '__main__':
    faces = detect_faces(pointcloud_np, confidence_threshold=0.5, draw_boxes=False)
    print(faces)

    f_box = detect_faces(pointcloud_np, confidence_threshold=0.5, draw_boxes=True)
    cv2.imwrite("test.png", f_box)


    mask = segment_all(depth_tensor, faces, depth_np, span=1)
    print(mask, mask.shape, bool(torch.any(mask != 0)), mask.max().item())
    mask_array = mask.mul(255).byte().cpu().numpy()
    img = Image.fromarray(mask_array)
    img.save("test_segment.png")
