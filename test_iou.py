import cv2
import numpy as np
from pathlib import Path

# Test loading and comparing two masks
mask1_path = 'results/sample/temp_files/chunks/chunk_0/masks/player/object_0.mp4'
mask2_path = 'results/sample/temp_files/chunks/chunk_1/masks/player/object_0.mp4'

def load_last_frame(path):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (frame > 0).astype(np.uint8)
    return None

def load_first_frame(path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return (frame > 0).astype(np.uint8)
    return None

def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / (union + 1e-6)

# Test chunk 0 -> 1
print("Testing chunk_0 -> chunk_1 for player prompt:")
for i in range(6):  # chunk_0 has 6 player objects
    mask1_p = f'results/sample/temp_files/chunks/chunk_0/masks/player/object_{i}.mp4'
    if not Path(mask1_p).exists():
        continue
    mask1 = load_last_frame(mask1_p)
    
    best_iou = 0
    best_j = -1
    for j in range(4):  # chunk_1 has 4 player objects
        mask2_p = f'results/sample/temp_files/chunks/chunk_1/masks/player/object_{j}.mp4'
        if not Path(mask2_p).exists():
            continue
        mask2 = load_first_frame(mask2_p)
        
        if mask1 is not None and mask2 is not None:
            iou = compute_iou(mask1, mask2)
            if iou > best_iou:
                best_iou = iou
                best_j = j
            print(f"  obj_{i} vs obj_{j}: IoU = {iou:.4f}")
    
    print(f"  -> Best match for obj_{i}: obj_{best_j} with IoU={best_iou:.4f}")
    print()
