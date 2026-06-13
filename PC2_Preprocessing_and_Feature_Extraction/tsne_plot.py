# tsne_plot.py
import numpy as np
import os, glob, cv2
from PIL import Image
from skimage import exposure, feature
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- CONFIG ---
PANCREAS_DATA_DIRECTORY = './Pancreas'
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400
CLIP_LIMIT = 0.01
KERNEL_SIZE = (80, 80)
CENTER_CROP_SIZE = 300
HU_RANGE_MIN = -200
HU_RANGE_MAX = 200


# --- Windowing ---
def apply_window(image_data, window_level, window_width):
    sim_hu = (image_data / 255.0) * (HU_RANGE_MAX - HU_RANGE_MIN) + HU_RANGE_MIN
    HU_min = window_level - (window_width / 2)
    HU_max = window_level + (window_width / 2)
    windowed_data = np.clip(sim_hu, HU_min, HU_max)
    scaled_data = (windowed_data - HU_min) / (HU_max - HU_min)
    return (scaled_data * 255).astype(np.uint8)


# --- Preprocess (Window + CLAHE + Center Crop) ---
def preprocess_and_roi(image_path, crop_size=CENTER_CROP_SIZE):
    try:
        img_8bit = np.array(Image.open(image_path).convert('L'))
        H, W = img_8bit.shape
        windowed = apply_window(img_8bit, WINDOW_LEVEL, WINDOW_WIDTH)
        float_img = windowed / 255.0
        clahe = exposure.equalize_adapthist(float_img, clip_limit=CLIP_LIMIT, kernel_size=KERNEL_SIZE)
        clahe_8bit = (clahe * 255).astype(np.uint8)

        start_h = H // 2 - crop_size // 2
        end_h = H // 2 + crop_size // 2
        start_w = W // 2 - crop_size // 2
        end_w = W // 2 + crop_size // 2
        if start_h < 0 or start_w < 0 or end_h > H or end_w > W:
            return np.array([])
        return clahe_8bit[start_h:end_h, start_w:end_w]
    except Exception:
        return np.array([])


# --- Feature extractors ---
def extract_lbp(roi):
    RADIUS, N_POINTS = 3, 24
    lbp = feature.local_binary_pattern(roi, N_POINTS, RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
    return hist


def extract_glcm(roi):
    glcm_input = (roi // (256 / 16)).astype(np.uint8)
    glcm = feature.graycomatrix(glcm_input, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16, symmetric=True, normed=True)
    return np.array([
        feature.graycoprops(glcm, 'contrast').mean(),
        feature.graycoprops(glcm, 'energy').mean(),
        feature.graycoprops(glcm, 'homogeneity').mean()
    ])


def extract_fourier(roi, num_rings=4):
    roi_f = roi.astype(np.float64)
    f = np.fft.fftshift(np.fft.fft2(roi_f))
    mag = np.abs(f)**2
    rows, cols = mag.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    r = np.sqrt(x*x + y*y)
    ring_radii = np.linspace(0, roi.shape[0]/2, num_rings+1)
    feats = []
    for i in range(num_rings):
        mask = np.logical_and(r >= ring_radii[i], r < ring_radii[i+1])
        vals = mag[mask]
        if vals.size > 0:
            energy, mean = np.sum(vals), np.mean(vals)
            prob = vals / np.sum(vals)
            ent = entropy(prob + 1e-10)
            feats.extend([energy, mean, ent])
        else:
            feats.extend([0,0,0])
    return np.array(feats)


def extract_hog(roi):
    roi_f = roi / 255.0
    hog_feats = feature.hog(roi_f, orientations=9, pixels_per_cell=(100,100),
                            cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
    return hog_feats


# --- Compile dataset features ---
def compile_features(base_dir):
    feature_list, labels = [], []
    class_map = {'NPancreas': 0, 'PPancreas': 1}

    for cname, lbl in class_map.items():
        files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG'):
            files.extend(glob.glob(os.path.join(base_dir, cname, ext)))
        print(f"{cname}: {len(files)} images")

        for f in files:
            roi = preprocess_and_roi(f)
            if roi.size == 0: continue
            try:
                feats = np.concatenate([
                    extract_lbp(roi),
                    extract_glcm(roi),
                    extract_fourier(roi),
                    extract_hog(roi)
                ])
                feature_list.append(feats)
                labels.append(lbl)
            except Exception:
                continue

    X = np.array(feature_list)
    y = np.array(labels)
    print(f"✅ Extracted {len(y)} samples | Feature matrix: {X.shape}")
    return X, y


# --- Run and visualize t-SNE ---
if __name__ == '__main__':
    X_raw, y = compile_features(PANCREAS_DATA_DIRECTORY)
    if X_raw.size == 0:
        print("❌ No features extracted — check dataset path and image sizes.")
        exit()

    X_scaled = StandardScaler().fit_transform(X_raw)
    print("Running t-SNE projection...")
    X_emb = TSNE(n_components=2, random_state=42, perplexity=20).fit_transform(X_scaled)

    plt.figure(figsize=(7,6))
    scatter = plt.scatter(X_emb[:,0], X_emb[:,1], c=y, cmap='coolwarm', s=25, alpha=0.8)
    plt.title("t-SNE Projection of Handcrafted Features (LBP + GLCM + Fourier + HOG)")
    plt.colorbar(scatter, ticks=[0,1], label='0 = Normal | 1 = Acute Pancreatitis')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(alpha=0.3)

    output_path = "tsne_clusters.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ t-SNE plot saved successfully as {output_path}")

