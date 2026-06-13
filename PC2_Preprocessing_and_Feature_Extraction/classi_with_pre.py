import numpy as np
import os
import glob
import cv2
from skimage import exposure, feature
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# --- FULL PIPELINE FUNCTION (RE-DEFINED) ---

def extract_lbp_features(image_path):
    """
    Applies the full preprocessing pipeline (Windowing -> CLAHE -> Bilateral Filter)
    and calculates the Uniform Local Binary Pattern (LBP) histogram features
    over a 300x300 center ROI.
    """
    # --- Parameters (Consistent with previous analysis) ---
    WINDOW_LEVEL = 40
    WINDOW_WIDTH = 400
    CLIP_LIMIT = 0.01
    KERNEL_SIZE = (80, 80)
    DIAMETER = 9
    SIGMA_COLOR = 75
    SIGMA_SPACE = 75
    CENTER_CROP_SIZE = 300
    RADIUS = 3
    N_POINTS = 8 * RADIUS
    METHOD = 'uniform'
    
    def apply_window(image_data, window_level, window_width):
        HU_RANGE_MIN = -200
        HU_RANGE_MAX = 200
        sim_hu = (image_data / 255.0) * (HU_RANGE_MAX - HU_RANGE_MIN) + HU_RANGE_MIN
        HU_min = window_level - (window_width / 2)
        HU_max = window_level + (window_width / 2)
        windowed_data = np.clip(sim_hu, HU_min, HU_max)
        scaled_data = (windowed_data - HU_min) / (HU_max - HU_min)
        scaled_data = (scaled_data * 255).astype(np.uint8)
        return scaled_data

    try:
        img_8bit = np.array(Image.open(image_path).convert('L'))
        H, W = img_8bit.shape
        
        # Preprocessing
        windowed = apply_window(img_8bit, WINDOW_LEVEL, WINDOW_WIDTH)
        float_img = windowed / 255.0
        clahe = exposure.equalize_adapthist(float_img, clip_limit=CLIP_LIMIT, kernel_size=KERNEL_SIZE)
        clahe_8bit = (clahe * 255).astype(np.uint8)
        final_img = cv2.bilateralFilter(clahe_8bit, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)
        
        # ROI Extraction
        start_h = H // 2 - CENTER_CROP_SIZE // 2
        end_h = H // 2 + CENTER_CROP_SIZE // 2
        start_w = W // 2 - CENTER_CROP_SIZE // 2
        end_w = W // 2 + CENTER_CROP_SIZE // 2
        roi = final_img[start_h:end_h, start_w:end_w]
        
        # LBP Feature Calculation
        lbp = feature.local_binary_pattern(roi, P=N_POINTS, R=RADIUS, method=METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
        
        return hist
        
    except Exception as e:
        # We skip files that cannot be processed (e.g., non-image, corrupted, or identifier files)
        return np.array([])


# --- FEATURE COMPILATION FUNCTION ---

def compile_dataset_features(base_data_dir):
    """
    Compiles LBP features from the NPancreas and PPancreas subdirectories.
    """
    feature_list = []
    label_list = []
    
    # Class names and corresponding labels based on your structure
    class_map = {'NPancreas': 0, 'PPancreas': 1}
    
    print(f"Starting feature extraction from: {base_data_dir}")

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
            features = extract_lbp_features(file_path)
            
            if features.size > 0:
                feature_list.append(features)
                label_list.append(label)
                count += 1
                
        print(f"Successfully extracted features from {count} images in '{class_name}'.")

    X = np.array(feature_list)
    y = np.array(label_list)
    
    print("\n--- Feature Compilation Complete ---")
    print(f"Total Images Processed: {len(y)}")
    print(f"Feature Matrix X Shape: {X.shape}")
    print(f"Label Vector y Shape: {y.shape}")
    
    return X, y


# --- CLASSIFICATION WORKFLOW ---

if __name__ == '__main__':
    # 1. DEFINE YOUR DATA DIRECTORY
    # REPLACE THIS with the actual path to your 'Pancreas' folder
    PANCREAS_DATA_DIRECTORY = './Pancreas' 
    
    # 2. Extract Features
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
        # We use a linear kernel for simplicity and interpretability
        classifier = SVC(kernel='linear', C=1.0, random_state=42) 
        classifier.fit(X_train, y_train)
        print("Training complete.")

        # 6. Evaluate on Test Set
        y_pred = classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n--- Classification Results ---")
        print(f"Classifier Used: Support Vector Machine (Linear Kernel)")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report (Precision, Recall, F1-Score):")
        
        # Use target_names to make the report readable
        target_names = ['Normal (0)', 'Pancreatitis (1)']
        print(classification_report(y_test, y_pred, target_names=target_names))