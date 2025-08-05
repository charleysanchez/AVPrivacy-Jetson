import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Set up output directory
output_dir = "saved_frames"
os.makedirs(output_dir, exist_ok=True)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
pipeline.start()

try:
    for i in range(30):  # Capture 30 frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Save as PNG or JPG using OpenCV
        filename = os.path.join(output_dir, f"frame_{i:03d}.png")  # or use .jpg
        cv2.imwrite(filename, color_image)

        depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
        depth_filename = os.path.join(output_dir, f"depth_frame_{i:03d}.png")
        cv2.imwrite(depth_filename, depth_vis)

        print(f"Saved {filename}")

finally:
    pipeline.stop()
