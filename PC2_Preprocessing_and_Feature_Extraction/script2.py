import numpy as np
import os
import glob
import cv2
from skimage import exposure, feature
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# Added confusion_matrix import
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
from scipy.stats import entropy # Needed for fourier_features if used

# --- MODIFIED LBP FUNCTION (PREPROCESSING REMOVED) --

def extract_lbp_features(image_path):
    """
    Calculates Uniform Local Binary Pattern (LBP) histogram features
    over a 300x300 center ROI, skipping all preprocessing steps.
    """
    # --- Parameters (Consistent with previous analysis) ---
    CENTER_CROP_SIZE = 300
    RADIUS = 3
    N_POINTS = 8 * RADIUS
    METHOD = 'uniform'
    
    try:
        # 1. Image Reading (Convert to Grayscale, 8-bit)
        img_8bit = np.array(Image.open(image_path).convert('L'))
        H, W = img_8bit.shape
        
        # NOTE: Skipping Windowing, CLAHE, and Bilateral Filter.
        
        # 2. ROI Extraction
        start_h = H // 2 - CENTER_CROP_SIZE // 2
        end_h = H // 2 + CENTER_CROP_SIZE // 2
        start_w = W // 2 - CENTER_CROP_SIZE // 2
        end_w = W // 2 + CENTER_CROP_SIZE // 2
        roi = img_8bit[start_h:end_h, start_w:end_w]
        
        # 3. LBP Feature Calculation
        lbp = feature.local_binary_pattern(roi, P=N_POINTS, R=RADIUS, method=METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
        
        return hist
        
    except Exception as e:
        # print(f"Skipping file {image_path} due to error: {e}") 
        return np.array([])


# --- PLACEHOLDER FUNCTIONS (Kept for compatibility) ---

def extract_glcm_features(image_path):
    """Placeholder for GLCM feature extraction."""
    # This function is not used when LBP is selected in compile_dataset_features
    return np.array([]) 

def extract_fourier_features(image_path):
    """Placeholder for Fourier feature extraction."""
    # This function is not used when LBP is selected in compile_dataset_features
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
            features = extract_lbp_features(file_path) # LBP is selected here!
            
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


# --- CLASSIFICATION WORKFLOW (MODIFIED FOR TP/FP/TN/FN DISPLAY) ---

if __name__ == '__main__':
    # 1. DEFINE YOUR DATA DIRECTORY
    # FIX: Using a raw string to handle the Windows path correctly.
    PANCREAS_DATA_DIRECTORY = r'D:\Assignments\MIA\PC2\Pancreas' 
    
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
        # C[i, j] is the number of observations known to be in group i and predicted to be in group j.
        # Target Names: Normal (0), Pancreatitis (1)
        cm = confusion_matrix(y_test, y_pred)
        
        # Extract individual values based on sklearn standard:
        # cm = [[TN, FP], [FN, TP]]
        TN = cm[0, 0] # True Negative (Normal predicted Normal)
        FP = cm[0, 1] # False Positive (Normal predicted Pancreatitis)
        FN = cm[1, 0] # False Negative (Pancreatitis predicted Normal)
        TP = cm[1, 1] # True Positive (Pancreatitis predicted Pancreatitis)

        accuracy = accuracy_score(y_test, y_pred) 
        
        # Calculate Normalized Metrics (Sensitivity/Recall and Specificity)
        # Normalized TP (Recall/Sensitivity) = TP / (TP + FN)
        # Normalized TN (Specificity) = TN / (TN + FP)
        
        # Handle division by zero for robustness
        total_actual_positive = TP + FN
        total_actual_negative = TN + FP
        
        recall = (TP / total_actual_positive) * 100 if total_actual_positive > 0 else 0.0
        specificity = (TN / total_actual_negative) * 100 if total_actual_negative > 0 else 0.0
        
        print("\n" + "="*70)
        print("        ✨ CLASSIFICATION METRICS & CONFUSION MATRIX (LBP) ✨")
        print("="*70)
        print(f"Classifier Used: Support Vector Machine (Linear Kernel)")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        
        print("\n--- Confusion Matrix (Raw Counts) ---")
        print(f"Actual/Predicted | {'Normal (0)':<15} | {'Pancreatitis (1)':<15}")
        print("-" * 70)
        # The TN and FP representation is clearly labeled here
        print(f"Normal (0)      | {TN:<15} <- *TN*   | {FP:<15} <- *FP*")
        # The FN and TP representation is clearly labeled here
        print(f"Pancreatitis (1) | {FN:<15} <- *FN*   | {TP:<15} <- *TP*")
        print("-" * 70)
        
        print(f"\nSummary of Normalized Metrics:")
        print(f"Normalized TP (Recall/Sensitivity): {recall:.2f}% (TP / Actual Positives)")
        print(f"Normalized TN (Specificity):      {specificity:.2f}% (TN / Actual Negatives)")
        
        print(f"\nSummary of Errors (Raw Counts):")
        print(f"True Positives (TP):   {TP} (Correctly predicted Pancreatitis)")
        print(f"True Negatives (TN):   {TN} (Correctly predicted Normal)")
        print(f"False Positives (FP):  {FP} (Incorrectly predicted Pancreatitis - Type I Error)")
        print(f"False Negatives (FN):  {FN} (Incorrectly predicted Normal - Type II Error)")
        
        # -----------------------------------------------------------
        ## 🎯 8. FEATURE IMPORTANCE ANALYSIS (SHOWING PERCENTAGE) 
        # -----------------------------------------------------------
        
        # Get the coefficients (weights) assigned to each feature.
        coefficients = classifier.coef_[0] 
        absolute_importance = np.abs(coefficients)
        total_importance = np.sum(absolute_importance)
        percentage_importance = (absolute_importance / total_importance) * 100
        sorted_indices = np.argsort(percentage_importance)[::-1]
        
        print("\n" + "="*85)
        print("    ✨ TOP 10 LBP FEATURE IMPORTANCE RANKING (PERCENTAGE CONTRIBUTION) ✨")
        print("="*85)
        print(f"{'Rank':<5} | {'LBP Feature Index (Bin)':<25} | {'Coefficient (Weight)':<20} | {'Percentage Importance':<20}")
        print("-" * 85)
        
        # Print the top 10 most important features
        for rank, i in enumerate(sorted_indices, 1):
            weight = coefficients[i]
            percent = percentage_importance[i]
            
            print(f"{rank:<5} | {i:<25} | {weight:<20.4f} | {percent:<19.2f}%")
            if rank >= 10:
                break
            
        print("-" * 85)
        print(f"Total features analyzed: {len(coefficients)}")
        
        # 9. Final Classification Report
        print("\nClassification Report (Precision, Recall, F1-Score):")
        target_names = ['Normal (0)', 'Pancreatitis (1)']
        print(classification_report(y_test, y_pred, target_names=target_names))