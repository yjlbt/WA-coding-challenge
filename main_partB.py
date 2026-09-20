import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install the required library first: pip install ultralytics")
    exit()

# Load the lightweight YOLOv8 model (downloads yolov8n.pt automatically on first run)
print("Loading YOLO object detection model...")
model = YOLO('yolov8n.pt')

# 1. Load traffic light 2D bounding box data (used as origin reference 0,0)
csv_path = "bbox_light.csv" if os.path.exists("bbox_light.csv") else "bboxes_light.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

ego_x, ego_y = [], []
cart_x, cart_y = [], []
valid_frames = []

print("Starting BEV trajectory processing...")

for idx, row in df.iterrows():
    frame_id = int(row['frame'])
    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
    
    # Skip invalid bounding boxes
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        continue

    npz_path = f"xyz/depth{frame_id:06d}.npz"
    rgb_path = f"rgb/left{frame_id:06d}.png"
    
    # Fallback to non-padded filenames if padded ones don't exist
    if not os.path.exists(npz_path):
        npz_path = f"xyz/depth{frame_id}.npz"
        rgb_path = f"rgb/left{frame_id}.png"

    if not os.path.exists(npz_path) or not os.path.exists(rgb_path):
        continue

    # ==========================================
    # Step 1: Calculate Ego vehicle position in the world coordinate system
    # ==========================================
    u_light, v_light = int((x1 + x2) / 2), int((y1 + y2) / 2)
    data = np.load(npz_path)
    points = data["points"] if "points" in data else data[data.files[0]]
    h_max, w_max = points.shape[0], points.shape[1]
    
    v_min, v_max = max(0, v_light - 1), min(h_max, v_light + 2)
    u_min, u_max = max(0, u_light - 1), min(w_max, u_light + 2)
    patch_light = points[v_min:v_max, u_min:u_max]
    
    valid_light_pts = patch_light[~np.isnan(patch_light).any(axis=-1)]
    valid_light_pts = valid_light_pts[(valid_light_pts != 0).any(axis=-1)]
    
    if len(valid_light_pts) == 0:
        continue
    
    light_xyz = np.mean(valid_light_pts, axis=0)
    X_l, Y_l = light_xyz[0], light_xyz[1]

    if not (np.isfinite(X_l) and np.isfinite(Y_l)):
        continue

    # Set Ego vehicle position in global coordinates
    ego_x_val = -X_l
    ego_y_val = -Y_l
    ego_x.append(ego_x_val)
    ego_y.append(ego_y_val)

    # ==========================================
    # Step 2: Use YOLO for precise golf cart localization
    # ==========================================
    rgb_img = cv2.imread(rgb_path)
    # Run YOLO inference (suppress console output per frame)
    results = model(rgb_img, verbose=False)
    
    best_cart_box = None
    max_area = 0
    
    # Parse YOLO results to find vehicles (COCO classes: 2=car, 5=bus, 7=truck)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in [2, 5, 7]:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0].cpu().numpy())
                area = (bx2 - bx1) * (by2 - by1)
                # Select the largest vehicle bounding box in the scene
                if area > max_area:
                    max_area = area
                    best_cart_box = [bx1, by1, bx2, by2]
    
    # Process depth data if YOLO successfully detected the vehicle
    if best_cart_box is not None:
        cx_u = int((best_cart_box[0] + best_cart_box[2]) / 2)
        cx_v = int((best_cart_box[1] + best_cart_box[3]) / 2)
        
        # Extract depth coordinates at the center of the bounding box
        c_v1, c_v2 = max(0, cx_v - 2), min(h_max, cx_v + 3)
        c_u1, c_u2 = max(0, cx_u - 2), min(w_max, cx_u + 3)
        patch_cart = points[c_v1:c_v2, c_u1:c_u2]
        
        v_pts = patch_cart[~np.isnan(patch_cart).any(axis=-1)]
        v_pts = v_pts[(v_pts != 0).any(axis=-1)]
        
        if len(v_pts) > 0:
            cart_xyz_cam = np.mean(v_pts, axis=0)
            X_c, Y_c = cart_xyz_cam[0], cart_xyz_cam[1]
            
            # World coordinates = Ego coordinates + Target relative coordinates
            cart_x_calc = ego_x_val + X_c
            cart_y_calc = ego_y_val + Y_c
            
            cart_x.append(cart_x_calc)
            cart_y.append(cart_y_calc)
        else:
            # Fallback if depth points are invalid
            cart_x.append(cart_x[-1] if len(cart_x) > 0 else ego_x_val + 5)
            cart_y.append(cart_y[-1] if len(cart_y) > 0 else ego_y_val)
    else:
        # Keep previous position if detection fails due to occlusion or false negatives
        cart_x.append(cart_x[-1] if len(cart_x) > 0 else ego_x_val + 5)
        cart_y.append(cart_y[-1] if len(cart_y) > 0 else ego_y_val)

    valid_frames.append(frame_id)

print(f"Successfully processed {len(ego_x)} trajectory frames!")

# ==========================================
# Step 3: Plotting and Video Generation
# ==========================================

# Filter out non-finite values to calculate clean axis limits
clean_x = np.array(ego_x + cart_x)[np.isfinite(ego_x + cart_x)]
clean_y = np.array(ego_y + cart_y)[np.isfinite(ego_y + cart_y)]
x_min, x_max = float(np.min(clean_x)) - 5, float(np.max(clean_x)) + 5
y_min, y_max = float(np.min(clean_y)) - 5, float(np.max(clean_y)) + 5

# --- Generate Static BEV Image ---
plt.figure(figsize=(8, 8))
plt.plot(ego_x, ego_y, color="tab:blue", linewidth=2, label="Ego Trajectory")
plt.plot(cart_x, cart_y, color="tab:orange", linestyle="--", linewidth=2, label="Golf Cart Path")

plt.plot(ego_x[0], ego_y[0], 'gs', label="Ego Start", markersize=8)
plt.plot(ego_x[-1], ego_y[-1], 'go', label="Ego Current", markersize=8)

plt.plot(cart_x[0], cart_y[0], 'r^', label="Cart Start", markersize=8)
plt.plot(cart_x[-1], cart_y[-1], 'ro', label="Cart Current", markersize=8)

plt.plot(0, 0, 'k*', label="Traffic Light (Origin)", markersize=12)

plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("Forward (X, m)")
plt.ylabel("Lateral (Y, m)")
plt.title("BEV Trajectories: Ego and Golf Cart (Ground Frame)")
plt.legend(loc="upper left")
plt.grid(True)
plt.axis("equal")

png_output = "bev_rich.png"
plt.savefig(png_output, dpi=300, bbox_inches='tight')
plt.show()
print(f"Exported static BEV image: {png_output}")

# --- Generate Dynamic BEV Video ---
mp4_output = "bev_rich.mp4"
fps = 10
frames_img = []

fig, ax = plt.subplots(figsize=(8, 8))

for i in range(len(ego_x)):
    ax.clear()
    
    # Plot incremental trajectories up to current frame 'i'
    ax.plot(ego_x[:i+1], ego_y[:i+1], color="tab:blue", linewidth=2, label="Ego Trajectory")
    ax.plot(ego_x[0], ego_y[0], 'gs', label="Ego Start", markersize=8)
    ax.plot(ego_x[i], ego_y[i], 'go', label="Ego Current", markersize=8)
    
    ax.plot(cart_x[:i+1], cart_y[:i+1], color="tab:orange", linestyle="--", linewidth=2, label="Golf Cart Path")
    ax.plot(cart_x[0], cart_y[0], 'r^', label="Cart Start", markersize=8)
    ax.plot(cart_x[i], cart_y[i], 'ro', label="Cart Current", markersize=8)
    
    ax.plot(0, 0, 'k*', label="Traffic Light (Origin)", markersize=12)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Forward (X, m)")
    ax.set_ylabel("Lateral (Y, m)")
    ax.set_title(f"BEV Trajectories: Ego & Golf Cart (Frame {valid_frames[i]})")
    ax.grid(True)
    ax.legend(loc="upper left")
    ax.set_aspect('equal')
    
    # Convert matplotlib figure to BGR image array for OpenCV
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    img_array = np.asarray(rgba_buffer)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    frames_img.append(img_bgr)

plt.close(fig)

# Write frames to MP4 video
height, width, _ = frames_img[0].shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(mp4_output, fourcc, fps, (width, height))

for frame in frames_img:
    out.write(frame)

out.release()
print(f"Exported dynamic BEV video: {mp4_output}")