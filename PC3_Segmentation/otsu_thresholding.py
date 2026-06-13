import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# ------------------------------------------------------------
# Step 1: Folder paths
# ------------------------------------------------------------
image_folder = r'archive/frames'   # Folder with CT images
mask_folder = r'archive/masks'     # Folder with ground truth masks

# Get all image paths
image_paths = sorted(glob(os.path.join(image_folder, '*.png')))
mask_paths = sorted(glob(os.path.join(mask_folder, '*.png')))

print(f"Found {len(image_paths)} CT images and {len(mask_paths)} masks.\n")

# ------------------------------------------------------------
# Step 2: Metric Functions
# ------------------------------------------------------------
def iou_score(y_true, y_pred):
    """Compute Intersection over Union (IoU)."""
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def dice_score(y_true, y_pred):
    """Compute Dice Coefficient."""
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
# Step 3: Loop over dataset and evaluate
# ------------------------------------------------------------
iou_scores, dice_scores = [], []
accs, precs, recs, f1s = [], [], [], []

for idx, (img_path, mask_path) in enumerate(zip(image_paths[:10], mask_paths[:10]), 1):
    # --- Load CT and mask ---
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or gt_mask is None:
        print(f"⚠️ Skipping unreadable pair: {img_path}")
        continue

    # --- Step 1: Preprocessing (Gaussian blur) ---
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # --- Step 2: Otsu thresholding ---
    thresh_value, otsu_img = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Step 3: Prepare ground truth ---
    gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]))
    gt_mask = (gt_mask > 128).astype(np.uint8)
    pred_mask = (otsu_img > 128).astype(np.uint8)

    # --- Step 4: Compute metrics ---
    iou = iou_score(gt_mask, pred_mask)
    dice = dice_score(gt_mask, pred_mask)
    TP, TN, FP, FN, acc, prec, rec, f1 = classification_metrics(gt_mask, pred_mask)

    # --- Store all metrics ---
    iou_scores.append(iou)
    dice_scores.append(dice)
    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)

    # --- Print per-image metrics ---
    print(f"[{idx}/{len(image_paths[:10])}] {os.path.basename(img_path)} ->")
    print(f" TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f" IoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}\n")

# ------------------------------------------------------------
# Step 4: Print Overall Metrics
# ------------------------------------------------------------
print("\n=== Overall Performance Across Dataset (First 10 images) ===")
print(f"Mean IoU:       {np.mean(iou_scores):.4f}")
print(f"Mean Dice:      {np.mean(dice_scores):.4f}")
print(f"Mean Accuracy:  {np.mean(accs):.4f}")
print(f"Mean Precision: {np.mean(precs):.4f}")
print(f"Mean Recall:    {np.mean(recs):.4f}")
print(f"Mean F1-Score:  {np.mean(f1s):.4f}")

# ------------------------------------------------------------
# Step 5: Visualization (optional)
# ------------------------------------------------------------
idx = 0  # View example
# img = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
# gt_mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
img = cv2.imread(r'archive\frames\bjorke_3.png', cv2.IMREAD_GRAYSCALE)
gt_mask = cv2.imread(r'archive\masks\bjorke_3.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)
_, otsu_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(18, 5))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray'); plt.title("Original CT"); plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(gt_mask, cmap='gray'); plt.title("Ground Truth Mask"); plt.axis('off')

plt.subplot(1, 4, 3)
plt.hist(img.ravel(), bins=256, color='gray')
plt.title("Histogram (Otsu Threshold)"); plt.axvline(_, color='r', linestyle='--'); plt.grid(False)

plt.subplot(1, 4, 4)
plt.imshow(otsu_img, cmap='gray'); plt.title("Predicted Mask (Otsu)"); plt.axis('off')

plt.tight_layout(); plt.show()
