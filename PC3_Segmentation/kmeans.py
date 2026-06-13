import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.cluster import KMeans

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
# Step 3: Loop over dataset
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

    # Resize and preprocess
    gt_mask = cv2.resize(gt_mask, (img.shape[1], img.shape[0]))
    gt_mask = (gt_mask > 128).astype(np.uint8)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # --- Step 1: Flatten image for clustering ---
    pixel_values = img_blur.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # --- Step 2: Apply K-Means ---
    K = 3  # number of clusters
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(pixel_values)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # --- Step 3: Reshape labels back to image ---
    segmented_image = labels.reshape(img.shape)

    # --- Step 4: Select cluster with highest mean intensity (lesion) ---
    lesion_cluster = np.argmax(centers)
    lesion_mask = (segmented_image == lesion_cluster).astype(np.uint8)

    # --- Step 5: Post-processing (clean up) ---
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    lesion_mask = (lesion_mask > 0).astype(np.uint8)

    # --- Step 6: Compute metrics ---
    iou = iou_score(gt_mask, lesion_mask)
    dice = dice_score(gt_mask, lesion_mask)
    TP, TN, FP, FN, acc, prec, rec, f1 = classification_metrics(gt_mask, lesion_mask)

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
# Step 4: Overall metrics
# ------------------------------------------------------------
print("\n=== Overall Performance (K-Means Segmentation — First 10 Images) ===")
print(f"Mean IoU:       {np.mean(iou_scores):.4f}")
print(f"Mean Dice:      {np.mean(dice_scores):.4f}")
print(f"Mean Accuracy:  {np.mean(accs):.4f}")
print(f"Mean Precision: {np.mean(precs):.4f}")
print(f"Mean Recall:    {np.mean(recs):.4f}")
print(f"Mean F1-Score:  {np.mean(f1s):.4f}")

# ------------------------------------------------------------
# Step 5: Visualization (example)
# ------------------------------------------------------------
idx = 100
img = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
gt_mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
gt_mask = (cv2.resize(gt_mask, (img.shape[1], img.shape[0])) > 128).astype(np.uint8)

# Apply K-Means again for visualization
pixel_values = img.reshape((-1, 1)).astype(np.float32)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(pixel_values)
segmented = kmeans.labels_.reshape(img.shape)
centers = kmeans.cluster_centers_
lesion_cluster = np.argmax(centers)
lesion_mask = (segmented == lesion_cluster).astype(np.uint8)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original CT")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(lesion_mask, cmap='gray')
plt.title("K-Means Predicted Lesion Mask")
plt.axis('off')

plt.tight_layout()
plt.show()
