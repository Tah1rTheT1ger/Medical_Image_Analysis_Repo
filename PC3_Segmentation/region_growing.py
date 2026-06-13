import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from collections import deque

# ------------------------------------------------------------
# Step 1: Dataset paths
# ------------------------------------------------------------
image_folder = r'archive/frames'
mask_folder = r'archive/masks'

image_paths = sorted(glob(os.path.join(image_folder, '*.png')))
mask_paths = sorted(glob(os.path.join(mask_folder, '*.png')))

print(f"Found {len(image_paths)} CT images and {len(mask_paths)} masks.\n")

# ------------------------------------------------------------
# Step 2: Metric functions
# ------------------------------------------------------------
def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    total = y_true.sum() + y_pred.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / total

def classification_metrics(y_true, y_pred):
    """Compute TP, TN, FP, FN, accuracy, precision, recall, f1."""
    TP = np.logical_and(y_true == 1, y_pred == 1).sum()
    TN = np.logical_and(y_true == 0, y_pred == 0).sum()
    FP = np.logical_and(y_true == 0, y_pred == 1).sum()
    FN = np.logical_and(y_true == 1, y_pred == 0).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return TP, TN, FP, FN, accuracy, precision, recall, f1

# ------------------------------------------------------------
# Step 3: Region Growing Function
# ------------------------------------------------------------
def region_growing(image, seed_point, threshold=15, connectivity=8):
    """Perform simple region growing from a seed point."""
    rows, cols = image.shape
    seed_intensity = image[seed_point]

    mask = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    q = deque([seed_point])
    visited[seed_point] = True

    # Define 4 or 8-connected neighbors
    if connectivity == 4:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while q:
        x, y = q.popleft()
        mask[x, y] = 1
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                if abs(int(image[nx, ny]) - int(seed_intensity)) < threshold:
                    q.append((nx, ny))
                    mask[nx, ny] = 1
                visited[nx, ny] = True
    return mask.astype(np.uint8)

# ------------------------------------------------------------
# Step 4: Loop through dataset
# ------------------------------------------------------------
iou_scores, dice_scores = [], []
accs, precs, recs, f1s = [], [], [], []

for idx, (img_path, mask_path) in enumerate(zip(image_paths[:10], mask_paths[:10]), 1):
    # --- Load CT and Ground Truth ---
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or gt_mask is None:
        print(f"⚠️ Skipping unreadable pair: {img_path}")
        continue

    # Resize GT to match image size
    gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]))
    gt_mask_bin = (gt_mask > 128).astype(np.uint8)

    # --- Find seed point from GT lesion area ---
    lesion_points = np.argwhere(gt_mask_bin == 1)
    if len(lesion_points) == 0:
        print(f"⚠️ No lesion found in GT for {os.path.basename(img_path)}, skipping.")
        continue
    # Randomly select one lesion pixel as seed (any valid pixel)
    seed_point = tuple(lesion_points[np.random.randint(0, len(lesion_points))])
    
    # --- Smooth image ---
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # --- Perform region growing ---
    region_mask = region_growing(img_blur, seed_point, threshold=15, connectivity=8)

    # --- Postprocess mask ---
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    region_mask = (region_mask > 0).astype(np.uint8)

    # --- Compute metrics ---
    iou = iou_score(gt_mask_bin, region_mask)
    dice = dice_score(gt_mask_bin, region_mask)
    TP, TN, FP, FN, acc, prec, rec, f1 = classification_metrics(gt_mask_bin, region_mask)

    # --- Store metrics ---
    iou_scores.append(iou)
    dice_scores.append(dice)
    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)

    # --- Print metrics per image ---
    print(f"[{idx}/{len(image_paths[:10])}] {os.path.basename(img_path)} ->")
    print(f" TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f" IoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n")

# ------------------------------------------------------------
# Step 5: Overall performance
# ------------------------------------------------------------
print("\n=== Overall Performance Across Dataset (First 10 Images) ===")
print(f"Mean IoU:       {np.mean(iou_scores):.4f}")
print(f"Mean Dice:      {np.mean(dice_scores):.4f}")
print(f"Mean Accuracy:  {np.mean(accs):.4f}")
print(f"Mean Precision: {np.mean(precs):.4f}")
print(f"Mean Recall:    {np.mean(recs):.4f}")
print(f"Mean F1-Score:  {np.mean(f1s):.4f}")

# ------------------------------------------------------------
# Step 6: Visualization for one example
# ------------------------------------------------------------
idx = 0
# img = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
# gt_mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
img = cv2.imread("archive/frames/bjorke_8.png", cv2.IMREAD_GRAYSCALE) # 1 and 3
gt_mask = cv2.imread("archive/masks/bjorke_8.png", cv2.IMREAD_GRAYSCALE)
gt_mask_bin = (cv2.resize(gt_mask, (img.shape[1], img.shape[0])) > 128).astype(np.uint8)

# Select seed from lesion in GT
lesion_points = np.argwhere(gt_mask_bin == 1)
seed_point = tuple(lesion_points[len(lesion_points)//2])  # middle lesion point

# Run region growing
img_blur = cv2.GaussianBlur(img, (5, 5), 0)
region_mask = region_growing(img_blur, seed_point, threshold=15)

# Visualize
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original CT Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth Mask (Lesion)")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(region_mask, cmap='gray')
plt.title("Region Growing Segmentation Result")
plt.axis('off')

plt.tight_layout()
plt.show()
