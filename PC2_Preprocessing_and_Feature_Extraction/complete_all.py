import numpy as np
import os
import glob
import cv2
from skimage import exposure, feature
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# Added missing imports for full functionality
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
from scipy.stats import entropy 

# --- CONSTANTS FOR ALL FUNCTIONS ---
WINDOW_LEVEL = 40
WINDOW_WIDTH = 400
CLIP_LIMIT = 0.01
KERNEL_SIZE = (80, 80)
DIAMETER = 9
SIGMA_COLOR = 75
SIGMA_SPACE = 75
CENTER_CROP_SIZE = 300
HU_RANGE_MIN = -200
HU_RANGE_MAX = 200

def apply_window(image_data, window_level, window_width):
    """Applies a custom HU window to the 8-bit image data."""
    # Simulate HU conversion (assuming 8-bit input from 0-255)
    sim_hu = (image_data / 255.0) * (HU_RANGE_MAX - HU_RANGE_MIN) + HU_RANGE_MIN
    HU_min = window_level - (window_width / 2)
    HU_max = window_level + (window_width / 2)
    
    # Apply window level/width
    windowed_data = np.clip(sim_hu, HU_min, HU_max)
    
    # Rescale back to 0-255 (8-bit)
    scaled_data = (windowed_data - HU_min) / (HU_max - HU_min)
    scaled_data = (scaled_data * 255).astype(np.uint8)
    return scaled_data

def preprocess_and_roi(image_path, crop_size=CENTER_CROP_SIZE):
    """Applies the full preprocessing pipeline (Windowing, CLAHE, Bilateral) and returns the central ROI."""
    try:
        # Load and convert to grayscale (L)
        img_8bit = np.array(Image.open(image_path).convert('L'))
        H, W = img_8bit.shape
        
        # Preprocessing: Windowing -> CLAHE -> Bilateral Filter
        windowed = apply_window(img_8bit, WINDOW_LEVEL, WINDOW_WIDTH)
        float_img = windowed / 255.0
        
        # CLAHE (Adaptive Histogram Equalization)
        clahe = exposure.equalize_adapthist(float_img, clip_limit=CLIP_LIMIT, kernel_size=KERNEL_SIZE)
        clahe_8bit = (clahe * 255).astype(np.uint8)
        
        # Bilateral Filter for noise reduction while preserving edges
        final_img = cv2.bilateralFilter(clahe_8bit, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)
        
        # ROI Extraction (Center Crop)
        start_h = H // 2 - crop_size // 2
        end_h = H // 2 + crop_size // 2
        start_w = W // 2 - crop_size // 2
        end_w = W // 2 + crop_size // 2
        
        # Check for valid crop dimensions
        if start_h < 0 or start_w < 0 or end_h > H or end_w > W:
             raise ValueError("Image is too small for the specified crop size.")

        roi = final_img[start_h:end_h, start_w:end_w]
        return roi
        
    except Exception as e:
        # print(f"Preprocessing error for {image_path}: {e}")
        return np.array([])


# --- INDIVIDUAL FEATURE EXTRACTION FUNCTIONS ---

def extract_lbp_features(image_path):
    """Extracts Uniform Local Binary Pattern (LBP) histogram features."""
    RADIUS = 3
    N_POINTS = 8 * RADIUS
    METHOD = 'uniform'
    
    roi = preprocess_and_roi(image_path)
    if roi.size == 0:
        return np.array([])

    try:
        # Compute LBP texture image
        lbp = feature.local_binary_pattern(roi, P=N_POINTS, R=RADIUS, method=METHOD)
        
        # Compute histogram of LBP
        n_bins = int(lbp.max() + 1)
        # Density=True normalizes the histogram
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins)) 
        return hist
    except Exception:
        return np.array([])

def extract_glcm_features(image_path):
    """Extracts GLCM features (Contrast, Energy, Homogeneity)."""
    GLCM_LEVELS = 16
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
    
    roi = preprocess_and_roi(image_path)
    if roi.size == 0:
        return np.array([])

    try:
        # Reduce the number of grey levels for GLCM computation
        glcm_input = (roi // (256 / GLCM_LEVELS)).astype(np.uint8)

        # Compute GLCM matrix
        glcm = feature.graycomatrix(
            glcm_input,
            distances=distances,
            angles=angles,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True
        )
        
        # Extract features and average them across all angles/distances
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        
        return np.array([contrast, energy, homogeneity])
    except Exception:
        return np.array([])

def extract_fourier_features(image_path):
    """Extracts Fourier-based spectral features (Energy, Mean, Entropy in 4 rings)."""
    NUM_RINGS = 4 
    RING_RADII = np.linspace(0, CENTER_CROP_SIZE / 2, NUM_RINGS + 1)
    
    roi = preprocess_and_roi(image_path)
    if roi.size == 0:
        return np.array([])

    try:
        roi_float = roi.astype(np.float64)
        f = np.fft.fft2(roi_float)
        fshift = np.fft.fftshift(f)
        # Calculate Power/Magnitude Spectrum
        magnitude_spectrum = np.abs(fshift)**2
        
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
        # Calculate radial distance from center
        r = np.sqrt(x*x + y*y)
        
        feature_vector = []
        # Iterate over the defined frequency rings
        for i in range(NUM_RINGS):
            r_min = RING_RADII[i]
            r_max = RING_RADII[i+1]
            # Create a mask for the current ring
            ring_mask = np.logical_and(r >= r_min, r < r_max)
            ring_values = magnitude_spectrum[ring_mask]
            
            if len(ring_values) > 0:
                ring_energy = np.sum(ring_values)
                ring_mean = np.mean(ring_values)
                # Calculate Entropy from normalized probability distribution
                ring_prob = ring_values / np.sum(ring_values)
                ring_entropy = entropy(ring_prob + 1e-10) 
                feature_vector.extend([ring_energy, ring_mean, ring_entropy])
            else:
                feature_vector.extend([0.0, 0.0, 0.0]) # 3 features per ring
        
        # Expected feature vector size: 4 rings * 3 features/ring = 12
        return np.array(feature_vector)
    except Exception:
        return np.array([])

def extract_hog_features(image_path):
    """
    Extracts Histogram of Oriented Gradients (HOG) features 
    on the preprocessed 300x300 ROI. 
    """
    # HOG Parameters for a (300/100)x(300/100) = 3x3 grid of cells, 
    # with (3-2+1)x(3-2+1) = 2x2 blocks of 2x2 cells each.
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (100, 100)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    roi = preprocess_and_roi(image_path)
    if roi.size == 0:
        return np.array([])

    try:
        # HOG expects floating point input
        roi_float = roi / 255.0
        
        # Calculate HOG features (Size: 2*2* (2*2*9) = 144)
        hog_features = feature.hog(
            roi_float, 
            orientations=HOG_ORIENTATIONS, 
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK, 
            transform_sqrt=False, # Use L2-Hys norm
            feature_vector=True, # Return the flattened vector
            block_norm='L2-Hys' 
        )
        
        return hog_features
    except Exception:
        return np.array([])


# --- FEATURE COMPILATION FUNCTION (MODIFIED) ---

def compile_dataset_features(base_data_dir):
    """
    MODIFIED: Compiles features by concatenating LBP, GLCM, Fourier, and HOG 
    features from the NPancreas and PPancreas subdirectories.
    """
    feature_list = []
    label_list = []
    
    # Class names and corresponding labels
    class_map = {'NPancreas': 0, 'PPancreas': 1}
    
    print(f"Starting combined feature extraction from: {base_data_dir}")

    for class_name, label in class_map.items():
        class_dir = os.path.join(base_data_dir, class_name)
        
        if not os.path.isdir(class_dir):
            print(f"Error: Directory not found at {class_dir}. Check path.")
            continue
            
        # Find all common image file types
        image_files = glob.glob(os.path.join(class_dir, '*.JPG')) + \
                      glob.glob(os.path.join(class_dir, '*.jpg')) + \
                      glob.glob(os.path.join(class_dir, '*.jpeg'))
        
        print(f"Found {len(image_files)} potential files in '{class_name}'.")

        count = 0
        for file_path in image_files:
            # --- 🚨 COMBINE ALL FOUR FEATURE METHODS HERE 🚨 ---
            try:
                # 1. Extract features individually
                lbp_features = extract_lbp_features(file_path) # ~26 features
                glcm_features = extract_glcm_features(file_path) # ~3 features
                fourier_features = extract_fourier_features(file_path) # ~12 features
                hog_features = extract_hog_features(file_path) # ~144 features
                
                # 2. Check if all features were extracted successfully (size > 0)
                if (lbp_features.size > 0 and 
                    glcm_features.size > 0 and 
                    fourier_features.size > 0 and 
                    hog_features.size > 0):
                    
                    # 3. Concatenate all features into one large vector (Feature Fusion)
                    features = np.concatenate([
                        lbp_features, 
                        glcm_features, 
                        fourier_features, 
                        hog_features
                    ])
                    
                    feature_list.append(features)
                    label_list.append(label)
                    count += 1
                else:
                    # print(f"Skipping {file_path}: one or more feature sets were empty.")
                    pass # Skip file if any feature extraction failed
                    
            except Exception as e:
                # print(f"Error combining features for {file_path}: {e}")
                pass # Continue to next file if combination fails
        # --------------------------------------------------------

        print(f"Successfully extracted features from {count} images in '{class_name}'.")

    X = np.array(feature_list)
    y = np.array(label_list)
    
    print("\n--- Feature Compilation Complete ---")
    print(f"Total Images Processed: {len(y)}")
    print(f"Feature Matrix X Shape: {X.shape}") # Should show (N, 185) where N is the number of images
    print(f"Label Vector y Shape: {y.shape}")
    
    return X, y


# --- CLASSIFICATION WORKFLOW ---

if __name__ == '__main__':
    # 1. DEFINE YOUR DATA DIRECTORY
    # FIX: Using a raw string to handle the Windows path correctly.
    PANCREAS_DATA_DIRECTORY = r'/mnt/d/Assignments/MIA/PC2/Pancreas' 
    
    # 2. Extract Features (Now combined)
    X_raw, y = compile_dataset_features(PANCREAS_DATA_DIRECTORY)
    
    if X_raw.size == 0:
        print("\nFATAL ERROR: No features were extracted. Check directory path and file formats.")
    else:
        # 3. Apply Feature Scaling (Crucial for SVM/Distance-based methods)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        # 4. Split Data into Training and Testing Sets (20% for testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42, stratify=y
        )
        
        print("\n--- Data Splitting Complete ---")
        print(f"Training Set Size: {X_train.shape[0]} samples")
        print(f"Testing Set Size: {X_test.shape[0]} samples")

        # 5. Train a Classifier (Support Vector Machine)
        print("\nTraining SVM Classifier...")
        # A linear kernel MUST be used for direct coefficient interpretation
        classifier = SVC(kernel='linear', C=1.0, random_state=42) 
        classifier.fit(X_train, y_train)
        print("Training complete.")
        
        # 6. Evaluate on Test Set
        y_pred = classifier.predict(X_test)
        
        # -----------------------------------------------------------
        ## 📊 7. DETAILED EVALUATION (CONFUSION MATRIX, TP, FP, TN, FN)
        # -----------------------------------------------------------
        
        # Calculate Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract individual values based on sklearn standard: cm = [[TN, FP], [FN, TP]]
        TN = cm[0, 0] # True Negative (Normal predicted Normal)
        FP = cm[0, 1] # False Positive (Normal predicted Pancreatitis)
        FN = cm[1, 0] # False Negative (Pancreatitis predicted Normal)
        TP = cm[1, 1] # True Positive (Pancreatitis predicted Pancreatitis)

        accuracy = accuracy_score(y_test, y_pred) 
        
        # Calculate Normalized Metrics (Sensitivity/Recall and Specificity)
        total_actual_positive = TP + FN
        total_actual_negative = TN + FP
        
        recall = (TP / total_actual_positive) * 100 if total_actual_positive > 0 else 0.0
        specificity = (TN / total_actual_negative) * 100 if total_actual_negative > 0 else 0.0
        
        print("\n" + "="*70)
        print("           ✨ CLASSIFICATION METRICS & CONFUSION MATRIX ✨")
        print("="*70)
        print(f"Classifier Used: Support Vector Machine (Linear Kernel)")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        
        print("\n--- Summary of Normalized Metrics ---")
        print(f"Normalized TP (Recall/Sensitivity): {recall:.2f}% (TP / Actual Positives)")
        print(f"Normalized TN (Specificity):        {specificity:.2f}% (TN / Actual Negatives)")
        
        print("\n--- Confusion Matrix (Raw Counts) ---")
        print(f"Actual/Predicted | {'Normal (0)':<15} | {'Pancreatitis (1)':<15}")
        print("-" * 70)
        print(f"Normal (0)       | {TN:<15} <- TN   | {FP:<15} <- FP")
        print(f"Pancreatitis (1) | {FN:<15} <- FN   | {TP:<15} <- TP")
        print("-" * 70)

        # -----------------------------------------------------------
        ## 🎯 8. FEATURE IMPORTANCE ANALYSIS 
        # -----------------------------------------------------------
        
        # # The feature label is generic now since the vector is a concatenation of 4 types
        # feature_label = 'Combined Feature Index' 

        # coefficients = classifier.coef_[0] 
        # absolute_importance = np.abs(coefficients)
        # total_importance = np.sum(absolute_importance)
        # percentage_importance = (absolute_importance / total_importance) * 100
        # sorted_indices = np.argsort(percentage_importance)[::-1]
        
        # print("\n" + "="*85)
        # print(f"      ✨ TOP 10 FEATURE IMPORTANCE RANKING (VECTOR SIZE: {len(coefficients)}) ✨")
        # print("="*85)
        # print(f"{'Rank':<5} | {feature_label:<25} | {'Coefficient (Weight)':<20} | {'Percentage Importance':<20}")
        # print("-" * 85)
        
        # # Print the top 10 most important features
        # for rank, i in enumerate(sorted_indices[:10], 1):
        #     weight = coefficients[i]
        #     percent = percentage_importance[i]
        #     print(f"{rank:<5} | {i:<25} | {weight:<20.4f} | {percent:<19.2f}%")
            
        # print("-" * 85)
        
        # 9. Final Classification Report
        print("\nClassification Report (Precision, Recall, F1-Score):")
        target_names = ['Normal (0)', 'Pancreatitis (1)']
        print(classification_report(y_test, y_pred, target_names=target_names))
