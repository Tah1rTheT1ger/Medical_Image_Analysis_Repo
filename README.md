# Medical Image Analysis: Computed Tomography (CT) Modality

This repository contains the coursework and implementation for the Medical Image Analysis (MIA) project, focusing on the Computed Tomography (CT) modality. The project spans across the entire pipeline of medical image analysis, from image acquisition and enhancement to segmentation and deep learning-based anomaly classification.

## Project Structure

The project is divided into multiple components (PCs), each addressing a fundamental stage of the medical image analysis workflow:

### 1. PC2: Preprocessing and Feature Extraction
This component addresses image restoration and enhancement for CT scans. The challenge with raw CT images includes low contrast and high noise.
- **Preprocessing:** Applied Windowing and Contrast Limited Adaptive Histogram Equalization (CLAHE) to significantly enhance clarity and reveal subtle features.
- **Feature Extraction:** Extracted highly discriminative handcrafted features to represent textual and structural properties:
  - Local Binary Pattern (LBP)
  - Gray Level Co-occurrence Matrix (GLCM)
  - Histogram of Oriented Gradients (HOG)
- **Classification:** Used these features to train traditional machine learning classifiers on a pancreas dataset to detect acute pancreatitis. 
  - **Support Vector Machine (SVM):** Achieved 95% accuracy on binary classification and 85% on multi-class.
  - **Random Forest (RF):** Achieved 98% accuracy on binary classification and 88% on multi-class.

### 2. PC3: Medical Image Segmentation
Focused on semantic segmentation to accurately delineate COVID-19 related pulmonary lesions in lung CT scans.
- **Traditional Methods:** Evaluated Otsu Thresholding, Region Growing, and K-Means clustering. Region Growing achieved the best traditional performance (Mean Accuracy: 0.9462).
- **Deep Learning Architectures:** Implemented advanced Encoder-Decoder paradigms for optimal lesion boundary detection.
  - **U-Net, Attention U-Net, and U-Net++:** Utilized skip connections to capture and preserve spatial detail.
  - **TransUNet & Swin-UNet:** Integrated Transformer blocks to capture both local details and global contextual relationships. 
  - **Best Performance:** Swin-UNet achieved the highest scores on the dataset with an Accuracy of 0.998 and an IoU of 0.851.

### 3. PC4: Anomaly Detection and Classification
This final component integrates foundational knowledge and feature engineering to classify lung tissue into five diagnostic categories (Normal, Benign, Adenocarcinoma, Squamous Cell Carcinoma, Large Cell Carcinoma).
- **Architectures:** Explored deep learning frameworks including Convolutional Neural Networks (CNN), ResNet50, and Swin Transformers.
- **Frameworks:** Evaluated both 1-Stage (direct 5-class prediction) and 2-Stage (Binary disease detection followed by specific carcinoma classification) approaches.
- **Results:** Highlighted the superiority of deep learning models in providing high confidence for complex carcinoma differentiation, demonstrating that optimal classification strategy depends heavily on architectural depth.

### 4. Paper Presentation: Review on Deep Learning in CT Image Analysis
A comprehensive review of state-of-the-art deep learning methods applied across the CT analysis pipeline.
- Explored domains like Low-Dose CT (LDCT) denoising (e.g., SGDNet, CTformer, SIST, CoreDiff).
- Discussed Super-Resolution (SR) and Image Generation techniques overcoming the photon-noise tradeoff while preserving diagnostic reliability and structural detail.

## Dataset References
- [1] Pancreas CT Images (Sichuan University Dataset)
- [2] COVID-19 CT Scan Lesion Segmentation Dataset (Merged public sources)
- [3] Lung Cancer CT Dataset (1,535 slices)

## Team
- Srivatsa Tarun Kondapalli (IMT2022034)
- Tahir Mohammed Khadarabad (IMT2022100)
- Venkata Sai Chaitanya Tadikonda (IMT2022545)
- Savithru Karthi Keyan Veerubothla (IMT2022569)

---
*This work was completed as part of the AIX 841 Medical Image Analysis course at IIIT-Bangalore.*
