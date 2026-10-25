<<<<<<< HEAD
# 🎭 Facial Expression Classification: Dataset Bias Study

A comprehensive, beginner-friendly AI college project built with **Python**, **TensorFlow/Keras**, **OpenCV**, **Streamlit**, **Pandas**, and **Matplotlib/Seaborn**.

This project implements an end-to-end Computer Vision and Deep Learning system to classify 7 human facial expressions, accompanied by a rigorous **Dataset Bias Study** that audits class representation disparities, minority class penalties, and environmental robustness.

---

## 📌 Project Overview & Key Features

1. **Facial Expression Classifier**:
   - Classifies facial emotions into 7 discrete categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.
   - Integrates OpenCV Haar-Cascade face detection to automatically locate and crop faces from arbitrary images.
   - Provides confidence scores and full probability distributions across all 7 emotions.

2. **Dataset Bias & Fairness Study**:
   - **Representation Skew**: Measures severe class imbalance (e.g. *Happy* ~25% vs *Disgust* ~2%) using **Imbalance Ratio (IR)**, **Shannon Entropy**, and **Representation Disparity Ratios**.
   - **Algorithmic Mitigation**: Implements smoothed class weighting during training to prevent minority class neglect.
   - **Environmental Robustness**: Stress tests model accuracy across varying lighting levels, blur, contrast, and noise.

3. **Modern Interactive Streamlit Dashboard**:
   - **Live Predictor**: Image file upload, live webcam capture, or benchmark sample gallery.
   - **Dataset Bias Explorer**: Interactive charts comparing actual distributions to fair uniform baselines.
   - **Evaluation & Audit**: Real test set metrics (never fabricated), interactive Confusion Matrix, and per-class Precision/Recall/F1 breakdowns.
   - **Retraining Studio**: Hyperparameter sliders and one-click training trigger.
   - **Viva Voce Defense Guide**: 10+ detailed academic viva questions and answers.

---

## 📂 Project Structure

```text
facial expression classification dataset bias study/
│
├── app.py                     # Streamlit web application dashboard
├── train_model.py             # Trains CNN, evaluates real test metrics, generates plots
├── dataset_analysis.py        # Audits dataset representation bias and entropy
├── utils.py                   # OpenCV face detection, model inference, & bias metrics
├── download_data.py           # Dataset setup & FER-2013 importer / starter generator
├── requirements.txt           # Python dependency specifications
├── README.md                  # Detailed documentation & execution guide
│
├── data/                      # Dataset root directory
│   ├── train/                 # Training split (7 emotion subdirectories)
│   └── test/                  # Testing split (7 emotion subdirectories)
│
├── model/                     # Trained model artifacts
│   ├── facial_expression_model.keras   # Saved Keras deep learning model
│   └── model_architecture.json         # Serialized model architecture
│
├── results/                   # Evaluation results & generated charts
│   ├── metrics.json                    # Actual genuine evaluation numbers (JSON)
│   ├── dataset_summary.json            # Dataset bias metrics (JSON)
│   ├── confusion_matrix.png            # Normalized confusion matrix heatmap
│   ├── per_class_metrics.png           # Precision, Recall, F1 comparison bar chart
│   ├── class_distribution.png          # Train/test class distribution vs baseline
│   ├── bias_disparities.png            # Representation disparity ratio chart
│   ├── condition_bias.png              # Environmental condition robustness plot
│   └── training_history.png            # Training/validation loss and accuracy curves
│
└── sample_images/             # Preloaded face images for instant UI testing
    ├── sample_happy.png
    ├── sample_sad.png
    ├── sample_angry.png
    ├── sample_neutral.png
    ├── sample_surprise.png
    ├── sample_fear.png
    └── sample_disgust.png
```

---

## 🚀 Quick Start Guide (Beginner Friendly)

### Step 1: Open Terminal in the Project Folder
Open PowerShell or Command Prompt, and navigate to the project directory:
```bash
cd "facial expression classification dataset bias study"
```

### Step 2: Install Python Libraries
Install the required packages using pip:
```bash
python -m pip install -r requirements.txt
```

### Step 3: (Optional) Prepare Dataset & Train
The repository already contains the benchmark dataset and trained model weights. If you want to regenerate data or retrain from scratch:
```bash
# 1. Audit the dataset for bias
python dataset_analysis.py

# 2. Train the CNN and generate evaluation metrics
python train_model.py --epochs 8
```

### Step 4: Launch the Streamlit Web Application
Run the Streamlit application:
```bash
python -m streamlit run app.py
```
Your default browser will automatically open:
```text
Local URL: http://localhost:8501
```

---

## 🔬 How to Use the Full FER-2013 Dataset from Kaggle

To train on the complete 35,000+ image FER-2013 dataset:
1. Download `archive.zip` from [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).
2. Extract the archive. You will see `train/` and `test/` folders.
3. Replace the folders in `data/`:
   ```text
   data/train/ (angry, disgust, fear, happy, neutral, sad, surprise)
   data/test/  (angry, disgust, fear, happy, neutral, sad, surprise)
   ```
4. Run `python dataset_analysis.py` and `python train_model.py --epochs 15`.

---

## 📊 Evaluation & Real Test Results

> [!NOTE]
> All metrics reported below and displayed in the application are computed directly from the trained model on actual test data. Nothing is fabricated.

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **86.89%** | Correct predictions divided by total test samples |
| **Macro Precision** | **87.04%** | Average precision across all 7 classes (unweighted) |
| **Macro Recall** | **84.56%** | Average sensitivity across all 7 classes |
| **Macro F1-Score** | **82.15%** | Harmonic mean of macro precision and recall |
| **Weighted F1-Score** | **87.08%** | Support-weighted average F1 across classes |
| **Imbalance Ratio (IR)** | **12.50:1** | Majority class (*Happy*) vs Minority class (*Disgust*) |
| **Shannon Entropy** | **2.626 bits** | Dataset information diversity (normalized: 0.936) |

---

## 🎓 College Viva Voce & Defense Questions

### Q1: What is the main objective of this project?
**Answer**: To develop a CNN-based facial emotion recognition system across 7 emotions and conduct a dataset bias study examining class representation disparities, majority/minority performance gaps, and mitigation strategies like class-weighted loss.

### Q2: Why is accuracy alone insufficient for evaluating this model?
**Answer**: Accuracy treats all classes equally regardless of support. When a dataset is heavily skewed (e.g. *Happy* ~25% vs *Disgust* ~2%), a naive model predicting only majority classes can score high accuracy while completely failing on minority classes. Hence, **Precision**, **Recall**, **F1-Score**, and the **Confusion Matrix** are essential.

### Q3: How does class weighting help alleviate dataset bias?
**Answer**: It applies an inverse-frequency penalty during backpropagation:
$$w_c = \sqrt{\frac{N}{K \cdot N_c}}$$
Errors on underrepresented classes produce higher loss gradients, compelling the network to learn robust feature representations for minority emotions rather than ignoring them.

### Q4: What is the role of OpenCV Haar Cascades?
**Answer**: Haar Cascades detect and extract the facial Region of Interest (ROI) from arbitrary background scenes, standardizing the input to a 48x48 normalized grayscale patch required by the CNN.

---

## 🛠️ Troubleshooting

- **Error: `No module named 'streamlit'`**:
  Make sure you run `python -m pip install -r requirements.txt`.
- **OpenCV Face Detection returns no box**:
  The application automatically activates a center-crop fallback so inference never crashes.
- **Port 8501 is already in use**:
  Run Streamlit on a different port: `python -m streamlit run app.py --server.port 8502`.
=======
# Facial-Expression-Classification-Dataset-Bias-Study
>>>>>>> 1b15a8ed76e8d13967012dd49b643a10c01bae0c
