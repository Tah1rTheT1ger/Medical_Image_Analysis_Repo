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

# --- FULL PIPELINE FUNCTION (RE-DEFINED) --

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

def extract_glcm_features(image_path):
    """
    Applies the full preprocessing pipeline and calculates GLCM features 
    (Contrast, Energy, Homogeneity) over a 300x300 center ROI.
    
    Returns: A 3-dimensional feature vector.
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
    
    # GLCM Parameters
    GLCM_LEVELS = 16
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees
    
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
        
        # 1. Preprocessing (Windowing -> CLAHE -> Bilateral Filter)
        windowed = apply_window(img_8bit, WINDOW_LEVEL, WINDOW_WIDTH)
        float_img = windowed / 255.0
        clahe = exposure.equalize_adapthist(float_img, clip_limit=CLIP_LIMIT, kernel_size=KERNEL_SIZE)
        clahe_8bit = (clahe * 255).astype(np.uint8)
        final_img = cv2.bilateralFilter(clahe_8bit, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)
        
        # 2. ROI Extraction
        start_h = H // 2 - CENTER_CROP_SIZE // 2
        end_h = H // 2 + CENTER_CROP_SIZE // 2
        start_w = W // 2 - CENTER_CROP_SIZE // 2
        end_w = W // 2 + CENTER_CROP_SIZE // 2
        roi = final_img[start_h:end_h, start_w:end_w]
        
        # 3. Prepare ROI for GLCM (Rescaling Intensities)
        # GLCM works better on a reduced number of gray levels
        glcm_input = (roi // (256 / GLCM_LEVELS)).astype(np.uint8)

        # 4. GLCM Feature Calculation
        glcm = feature.graycomatrix(
            glcm_input,
            distances=distances,
            angles=angles,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True
        )
        
        # Extract and average the three key features across all angles
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        energy = feature.graycoprops(glcm, 'energy').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        
        return np.array([contrast, energy, homogeneity])
        
    except Exception as e:
        # print(f"Skipping file {image_path} due to error: {e}") 
        return np.array([])

def extract_fourier_features(image_path):
    """
    Applies the full preprocessing pipeline and calculates Fourier-based spectral features 
    (Energy and Entropy in rings/wedges) over a 300x300 center ROI.
    
    Returns: A 12-dimensional feature vector (4 Rings * 3 Measures).
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
    
    # Fourier Parameters
    # Number of concentric rings for frequency analysis
    NUM_RINGS = 4 
    # The radii defining the rings (normalized to half the crop size)
    RING_RADII = np.linspace(0, CENTER_CROP_SIZE / 2, NUM_RINGS + 1)
    
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
        
        # 1. Preprocessing (Windowing -> CLAHE -> Bilateral Filter)
        windowed = apply_window(img_8bit, WINDOW_LEVEL, WINDOW_WIDTH)
        float_img = windowed / 255.0
        clahe = exposure.equalize_adapthist(float_img, clip_limit=CLIP_LIMIT, kernel_size=KERNEL_SIZE)
        clahe_8bit = (clahe * 255).astype(np.uint8)
        final_img = cv2.bilateralFilter(clahe_8bit, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)
        
        # 2. ROI Extraction
        start_h = H // 2 - CENTER_CROP_SIZE // 2
        end_h = H // 2 + CENTER_CROP_SIZE // 2
        start_w = W // 2 - CENTER_CROP_SIZE // 2
        end_w = W // 2 + CENTER_CROP_SIZE // 2
        roi = final_img[start_h:end_h, start_w:end_w].astype(np.float64)
        
        # 3. Fourier Transform
        # Apply 2D FFT to the ROI
        f = np.fft.fft2(roi)
        # Shift the zero-frequency component to the center for visualization/analysis
        fshift = np.fft.fftshift(f)
        # Calculate the Power Spectrum (Magnitude Squared)
        magnitude_spectrum = np.abs(fshift)**2
        
        # 4. Feature Extraction using Rings
        # Create a coordinate grid
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2
        y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
        # Calculate radial distance from center
        r = np.sqrt(x*x + y*y)
        
        feature_vector = []

        # Iterate through the defined rings to extract scale-invariant features
        for i in range(NUM_RINGS):
            r_min = RING_RADII[i]
            r_max = RING_RADII[i+1]
            
            # Create a mask for the current ring
            ring_mask = np.logical_and(r >= r_min, r < r_max)
            ring_values = magnitude_spectrum[ring_mask]
            
            if len(ring_values) > 0:
                # Feature 1: Energy (L2-norm squared, total power in the band)
                ring_energy = np.sum(ring_values)
                
                # Feature 2: Mean Magnitude
                ring_mean = np.mean(ring_values)
                
                # Feature 3: Entropy (measures spectral spread/randomness in the band)
                # Normalize values to a probability distribution for entropy calculation
                ring_prob = ring_values / np.sum(ring_values)
                # Use a small epsilon to avoid log(0)
                ring_entropy = entropy(ring_prob + 1e-10) 

                feature_vector.extend([ring_energy, ring_mean, ring_entropy])
            else:
                # Handle empty rings (unlikely with this setup, but good practice)
                feature_vector.extend([0.0, 0.0, 0.0])

        return np.array(feature_vector)
        
    except Exception as e:
        # print(f"Skipping file {image_path} due to error: {e}") 
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
            # features = extract_fourier_features(file_path)
            # features = extract_glcm_features(file_path)
            
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

# if _name_ == '_main_':
#     # 1. DEFINE YOUR DATA DIRECTORY
#     # REPLACE THIS with the actual path to your 'Pancreas' folder
#     PANCREAS_DATA_DIRECTORY = 'D:\Sem_7\Medical_Image_Analysis\Project\Pancreas CT images\Pancreas CT images\Pancreas' 
    
#     # 2. Extract Features
#     X_raw, y = compile_dataset_features(PANCREAS_DATA_DIRECTORY)
    
#     if X_raw.size == 0:
#         print("\nFATAL ERROR: No features were extracted. Check directory path and file formats.")
#     else:
#         # 3. Apply Feature Scaling (Crucial for SVM/Distance-based methods)
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X_raw)
        
#         # 4. Split Data into Training and Testing Sets (20% for testing)
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y, test_size=0.20, random_state=42, stratify=y
#         )
        
#         print("\n--- Data Splitting Complete ---")
#         print(f"Training Set Size: {X_train.shape[0]} samples")
#         print(f"Testing Set Size: {X_test.shape[0]} samples")

#         # 5. Train a Classifier (Support Vector Machine)
#         print("\nTraining SVM Classifier...")
#         # We use a linear kernel for simplicity and interpretability
#         classifier = SVC(kernel='linear', C=1.0, random_state=42) 
#         classifier.fit(X_train, y_train)
#         print("Training complete.")

#         # 6. Evaluate on Test Set
#         y_pred = classifier.predict(X_test)
        
#         accuracy = accuracy_score(y_test, y_pred)
        
#         print("\n--- Classification Results ---")
#         print(f"Classifier Used: Support Vector Machine (Linear Kernel)")
#         print(f"Overall Accuracy: {accuracy*100:.2f}%")
#         print("\nClassification Report (Precision, Recall, F1-Score):")
        
#         # Use target_names to make the report readable
#         target_names = ['Normal (0)', 'Pancreatitis (1)']
#         print(classification_report(y_test, y_pred, target_names=target_names))

if __name__ == '__main__':
    # 1. DEFINE YOUR DATA DIRECTORY
    # FIX: Using a raw string to handle the Windows path correctly.
    PANCREAS_DATA_DIRECTORY = r'D:\Assignments\MIA\PC2\Pancreas' 
    
    # 2. Extract Features
    # This relies on the LBP feature extraction function being defined above this block.
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
        
        # -----------------------------------------------------------
        ## 🎯 7. FEATURE IMPORTANCE ANALYSIS (SHOWING PERCENTAGE) 
        # -----------------------------------------------------------
        
        # Get the coefficients (weights) assigned to each of the 26 features.
        coefficients = classifier.coef_[0] 
        
        # Calculate the absolute importance magnitude
        absolute_importance = np.abs(coefficients)
        
        # Calculate the total absolute importance (sum of all feature contributions)
        total_importance = np.sum(absolute_importance)
        
        # Calculate the percentage importance for each feature
        # Percentage = (Feature's Absolute Weight / Total Absolute Weights) * 100
        percentage_importance = (absolute_importance / total_importance) * 100
        
        # Get the indices of the features sorted by percentage (highest first)
        sorted_indices = np.argsort(percentage_importance)[::-1]
        
        print("\n" + "="*85)
        print("    ✨ TOP 10 LBP FEATURE IMPORTANCE RANKING (PERCENTAGE CONTRIBUTION) ✨")
        print("="*85)
        print(f"{'Rank':<5} | {'LBP Feature Index (Bin)':<25} | {'Coefficient (Weight)':<20} | {'Percentage Importance':<20}")
        print("-" * 85)
        
        # Print the top 10 most important features
        # for rank, i in enumerate(sorted_indices[:10], 1):
        for rank, i in enumerate(sorted_indices, 1):
            weight = coefficients[i]
            percent = percentage_importance[i]
            
            # The coefficient's sign tells us the direction:
            # Positive weight -> Pushes prediction toward Pancreatitis (1)
            # Negative weight -> Pushes prediction toward Normal (0)
            
            print(f"{rank:<5} | {i:<25} | {weight:<20.4f} | {percent:<19.2f}%")
            
        print("-" * 85)
        print(f"Total features analyzed: {len(coefficients)}")
        
        # -----------------------------------------------------------
        ## 📊 8. EVALUATION (Existing Logic)
        # -----------------------------------------------------------
        
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