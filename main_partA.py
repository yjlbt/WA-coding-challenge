import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Load traffic light 2D bounding box data
csv_path = "bbox_light.csv"
if not os.path.exists(csv_path):
    csv_path = "bboxes_light.csv"

df = pd.read_csv(csv_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

ego_x = []
ego_y = []
frame_ids = []

print("Processing data and extracting ego-vehicle trajectory...")

# 2. Iterate through frames and compute 3D camera coordinates
for idx, row in df.iterrows():
    frame_id = int(row['frame'])
    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
    
    # Skip frames where bounding box is invalid [0, 0, 0, 0]
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        continue

    # Construct file path with 6-digit zero padding (e.g., xyz/depth000001.npz)
    npz_path = f"xyz/depth{frame_id:06d}.npz"
    if not os.path.exists(npz_path):
        # Fallback in case of non-zero-padded or other depth formats
        npz_path = f"xyz/depth{frame_id}.npz"

    if not os.path.exists(npz_path):
        print(f"Skipping missing frame: {npz_path}")
        continue

    # Calculate bounding box center pixel coordinates (u, v)
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)

    # Load point cloud data
    data = np.load(npz_path)
    
    # Adapt to key name inside npz file ('points' or 'xyz')
    if "points" in data:
        points = data["points"]
    else:
        points = data[data.files[0]]
    
    # Extract 3x3 patch around center pixel to reduce noise
    h_max, w_max = points.shape[0], points.shape[1]
    v_min, v_max = max(0, v - 1), min(h_max, v + 2)
    u_min, u_max = max(0, u - 1), min(w_max, u + 2)
    patch = points[v_min:v_max, u_min:u_max]
    
    # Filter out invalid depth points (NaNs and all zeros)
    valid_pts = patch[~np.isnan(patch).any(axis=-1)]
    valid_pts = valid_pts[(valid_pts != 0).any(axis=-1)]
    
    if len(valid_pts) > 0:
        cam_xyz = np.mean(valid_pts, axis=0) # [X_cam, Y_cam, Z_cam]
    else:
        cam_xyz = points[v, u]
        
    X_c, Y_c, Z_c = cam_xyz[0], cam_xyz[1], cam_xyz[2]
    
    # Check if point cloud coordinate at center is valid
    if np.isnan(X_c) or np.isnan(Y_c) or (X_c == 0 and Y_c == 0):
        continue

    # Invert camera relative positions to get ego-vehicle ground coordinates (Forward X, Lateral Y)
    ego_x.append(-X_c)
    ego_y.append(-Y_c)
    frame_ids.append(frame_id)

print(f"Successfully extracted coordinates for {len(ego_x)} frames!")

if len(ego_x) == 0:
    print("Error: No valid frame data extracted. Please check file paths.")
    exit()

# 3. Plot and save static trajectory image: trajectory.png
plt.figure(figsize=(8, 8))
plt.plot(ego_x, ego_y, label="Ego trajectory", color="tab:blue", linewidth=2)
plt.plot(ego_x[0], ego_y[0], 'rx', label="Start", markersize=10, markeredgewidth=2)
plt.plot(ego_x[-1], ego_y[-1], 'go', label="End", markersize=10)
plt.plot(0, 0, 'k*', label="Traffic light (origin)", markersize=12)

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel("Forward (X, m)")
plt.ylabel("Lateral (Y, m)")
plt.title("Ego-Vehicle Trajectory (BEV Ground Frame)")
plt.legend()
plt.grid(True)
plt.axis("equal")

png_output = "trajectory.png"
plt.savefig(png_output, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved static trajectory image: {png_output}")

# 4. Generate and save animation video: trajectory.mp4 using OpenCV
import cv2

print("Exporting trajectory animation video: trajectory.mp4...")

mp4_output = "trajectory.mp4"
fps = 10
fig_size = (8, 8)

frames_img = []
fig, ax = plt.subplots(figsize=fig_size)

for frame_idx in range(len(ego_x)):
    ax.clear()
    ax.plot(ego_x[:frame_idx+1], ego_y[:frame_idx+1], color="tab:blue", linewidth=2, label="Ego trajectory")
    ax.plot(ego_x[0], ego_y[0], 'rx', label="Start", markersize=10, markeredgewidth=2)
    if frame_idx > 0:
        ax.plot(ego_x[frame_idx], ego_y[frame_idx], 'go', label="Current Position", markersize=8)
    ax.plot(0, 0, 'k*', label="Traffic light (origin)", markersize=12)
    
    ax.set_xlim(min(ego_x) - 2, max(ego_x) + 2)
    ax.set_ylim(min(0, min(ego_y)) - 2, max(ego_y) + 2)
    ax.set_xlabel("Forward (X, m)")
    ax.set_ylabel("Lateral (Y, m)")
    ax.set_title(f"Ego-Vehicle Trajectory Animation (Frame {frame_ids[frame_idx]})")
    ax.grid(True)
    ax.legend(loc="upper right")
    ax.set_aspect('equal')
    
    # Force canvas redraw
    fig.canvas.draw()
    
    # Get RGBA buffer safely on modern Matplotlib versions
    rgba_buffer = fig.canvas.buffer_rgba()
    img_array = np.asarray(rgba_buffer)
    
    # Convert RGBA to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    frames_img.append(img_bgr)

plt.close(fig)

# Write frames to MP4 video file
height, width, _ = frames_img[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mp4_output, fourcc, fps, (width, height))

for frame in frames_img:
    out.write(frame)

out.release()
print(f"Successfully exported animation video: {mp4_output}")