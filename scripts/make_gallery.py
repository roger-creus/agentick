"""Create a gallery video grid of all tasks."""

import cv2
import numpy as np
import os
import glob

VIDEOS_DIR = "videos"
OUTPUT_PATH = "videos/gallery_all_tasks.mp4"

# Grid layout for all tasks
COLS = 8
ROWS = 5
CELL_W = 200  # width per cell (scaled down from 288)
CELL_H = 200  # height for video portion
LABEL_H = 24  # height for task name label
CELL_TOTAL_H = CELL_H + LABEL_H
FPS = 8
DURATION_S = 20  # seconds to keep the gallery running
PADDING = 2  # pixels between cells
BG_COLOR = (30, 30, 30)  # dark gray background
LABEL_BG = (20, 20, 20)
LABEL_FG = (255, 255, 255)

# Collect one video per task: easy + dense
video_files = sorted(glob.glob(os.path.join(VIDEOS_DIR, "*_easy_dense.mp4")))
print(f"Found {len(video_files)} task videos")

# Extract task names from filenames like "01_BacktrackPuzzle_easy_dense.mp4"
task_names = []
for f in video_files:
    base = os.path.basename(f).replace("_easy_dense.mp4", "")
    # Remove leading number and underscore
    name = base.split("_", 1)[1] if "_" in base else base
    # Add spaces before capitals for readability
    readable = ""
    for i, c in enumerate(name):
        if c.isupper() and i > 0 and name[i - 1].islower():
            readable += " "
        readable += c
    task_names.append(readable)

# Open all video captures and pre-read all frames
print("Reading video frames...")
all_frames = []
for i, vf in enumerate(video_files):
    cap = cv2.VideoCapture(vf)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to cell dimensions
        frame = cv2.resize(frame, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        # Blank fallback
        frames = [np.full((CELL_H, CELL_W, 3), 50, dtype=np.uint8)]
    all_frames.append(frames)
    print(f"  [{i+1:2d}/38] {task_names[i]}: {len(frames)} frames")

# Create label images for each task
labels = []
for name in task_names:
    label_img = np.full((LABEL_H, CELL_W, 3), LABEL_BG[0], dtype=np.uint8)
    # Pick font size to fit
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.35
    thickness = 1
    (tw, th), _ = cv2.getTextSize(name, font, scale, thickness)
    # Shrink if needed
    while tw > CELL_W - 8 and scale > 0.2:
        scale -= 0.02
        (tw, th), _ = cv2.getTextSize(name, font, scale, thickness)
    x = (CELL_W - tw) // 2
    y = (LABEL_H + th) // 2
    cv2.putText(label_img, name, (x, y), font, scale, LABEL_FG, thickness, cv2.LINE_AA)
    labels.append(label_img)

# Canvas size
canvas_w = COLS * CELL_W + (COLS + 1) * PADDING
canvas_h = ROWS * CELL_TOTAL_H + (ROWS + 1) * PADDING

total_frames = DURATION_S * FPS
print(f"\nOutput: {canvas_w}x{canvas_h} @ {FPS}fps, {total_frames} frames ({DURATION_S}s)")

# Write video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (canvas_w, canvas_h))

for frame_idx in range(total_frames):
    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR[0], dtype=np.uint8)

    for task_idx in range(len(video_files)):
        row = task_idx // COLS
        col = task_idx % COLS

        # Get frame (loop if needed)
        task_frames = all_frames[task_idx]
        f = task_frames[frame_idx % len(task_frames)]

        # Position on canvas
        x = PADDING + col * (CELL_W + PADDING)
        y = PADDING + row * (CELL_TOTAL_H + PADDING)

        # Place label
        canvas[y : y + LABEL_H, x : x + CELL_W] = labels[task_idx]
        # Place video frame
        canvas[y + LABEL_H : y + LABEL_H + CELL_H, x : x + CELL_W] = f

    writer.write(canvas)

    if (frame_idx + 1) % (FPS * 5) == 0:
        print(f"  Written {frame_idx + 1}/{total_frames} frames...")

writer.release()

# Re-encode with ffmpeg for better compatibility (h264)
final_path = OUTPUT_PATH.replace(".mp4", "_h264.mp4")
os.system(
    f'ffmpeg -y -i "{OUTPUT_PATH}" -c:v libx264 -pix_fmt yuv420p -crf 20 "{final_path}" 2>/dev/null'
)
if os.path.exists(final_path):
    os.replace(final_path, OUTPUT_PATH)
    print(f"\nDone! Gallery saved to: {OUTPUT_PATH}")
else:
    print(f"\nDone (raw mp4v)! Gallery saved to: {OUTPUT_PATH}")

file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"File size: {file_size_mb:.1f} MB")
