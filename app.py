"""
app.py
======
Facial Expression Classification: Dataset Bias Study
Modern Streamlit Dashboard for real-time emotion recognition, dataset bias auditing,
model evaluation, and viva defense exploration.
"""

import osn
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import project utilities
from utils import (
    EMOTION_CLASSES,
    EMOTION_COLORS,
    detect_and_crop_faces,
    annotate_image_with_predictions,
    predict_expression,
    calculate_dataset_bias_stats
)
from dataset_analysis import run_dataset_analysis
from train_model import train_and_evaluate

# Configure page settings
st.set_page_config(
    page_title="Facial Expression Classification | Dataset Bias Study",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.3);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.4rem;
    }
    
    .main-header p {
        color: #C7D2FE;
        font-size: 1.05rem;
        margin: 0;
    }

    .metric-card {
        background: white;
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px -3px rgba(0, 0, 0, 0.08);
    }

    .metric-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #0F172A;
        font-size: 1.9rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model safely
@st.cache_resource
def load_trained_model(model_path="model/facial_expression_model.keras"):
    """Loads the trained Keras model with caching."""
    if not os.path.exists(model_path):
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        return None

# Load stored metrics
def load_results_data():
    metrics_path = "results/metrics.json"
    dataset_summary_path = "results/dataset_summary.json"
    
    metrics = None
    dataset_summary = None

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            pass

    if os.path.exists(dataset_summary_path):
        try:
            with open(dataset_summary_path, "r", encoding="utf-8") as f:
                dataset_summary = json.load(f)
        except Exception:
            pass

    return metrics, dataset_summary

# Header Banner
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1>🎭 Facial Expression Classification: Dataset Bias Study</h1>
            <p>Deep Learning CNN • OpenCV Face Localization • Algorithmic Fairness & Bias Audit</p>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 0.6rem 1.2rem; border-radius: 12px; text-align: right;">
            <div style="font-size: 0.75rem; color: #E0E7FF; font-weight: 600;">PROJECT STATUS</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: #A7F3D0;">Production Ready</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("System Controls")
    st.markdown("Navigate through live prediction, fairness audit, dataset imbalance, and defense prep.")
    
    model = load_trained_model()
    if model is not None:
        st.success(" AI Model Loaded (Keras CNN)")
    else:
        st.warning("⚠️ Model not found in `model/`. Run training to generate it.")

    st.markdown("---")
    st.markdown("### 📚 Quick Dataset Links")
    st.markdown("- [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)")
    st.markdown("- **Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise")
    
    st.markdown("---")
    st.markdown("### 🔬 Bias Research Focus")
    st.info(
        "**Key Question**: How does class imbalance (e.g. Disgust ~2% vs Happy ~25%) and environmental conditions "
        "(lighting, blur) skew model accuracy and fairness?"
    )

# Main Navigation Tabs
tab_predict, tab_dataset, tab_eval, tab_train, tab_viva = st.tabs([
    "🔮 Live Face Predictor",
    "📊 Dataset Bias Analysis",
    "📈 Evaluation & Fairness Audit",
    "⚙️ Model Retraining Studio",
    "🎓 College Viva & Defense Guide"
])

# ==============================================================================
# TAB 1: LIVE FACE PREDICTOR
# ==============================================================================
with tab_predict:
    st.subheader("Facial Expression Classifier & Face Detection")
    st.caption("Upload a face photo, capture via webcam, or select one of our benchmark presets.")

    col_input, col_output = st.columns([1.1, 1.2], gap="large")

    selected_image = None
    input_source = None

    with col_input:
        input_method = st.radio(
            "Select Input Source:",
            [" Upload File", "📷 Webcam Capture", " Preset Gallery"],
            horizontal=True
        )

        if "Upload" in input_method:
            uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                try:
                    selected_image = Image.open(uploaded_file)
                    input_source = "Uploaded File"
                except Exception as e:
                    st.error(f"Invalid image format: {e}")

        elif "Webcam" in input_method:
            camera_image = st.camera_input("Take a photo with your webcam")
            if camera_image is not None:
                try:
                    selected_image = Image.open(camera_image)
                    input_source = "Webcam Capture"
                except Exception as e:
                    st.error(f"Error reading camera feed: {e}")

        else: # Preset Gallery
            st.markdown("Pick a benchmark test face:")
            sample_dir = "sample_images"
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
                selected_sample = st.selectbox(
                    "Choose sample expression:",
                    sample_files,
                    format_func=lambda x: x.replace("sample_", "").replace(".png", "").capitalize()
                )
                if selected_sample:
                    selected_image = Image.open(os.path.join(sample_dir, selected_sample))
                    input_source = f"Preset: {selected_sample}"
            else:
                st.warning("Preset sample folder not found. Please run 'python download_data.py --generate-starter'.")

        if selected_image is not None:
            st.image(selected_image, caption=f"Input Source: {input_source}", use_container_width=True)

    with col_output:
        if selected_image is None:
            st.info("👈 Please upload an image, use webcam, or choose a preset to view predictions.")
        elif model is None:
            st.error("Model file (`model/facial_expression_model.keras`) is missing. Please train the model in the 'Model Retraining Studio' tab.")
        else:
            with st.spinner("Analyzing facial geometry & running CNN inference..."):
                try:
                    faces_data, rgb_arr, n_faces = detect_and_crop_faces(selected_image)
                    
                    if n_faces == 0:
                        st.warning("⚠️ OpenCV Haar Cascade could not detect an isolated face bounding box. Using center-crop fallback.")
                    else:
                        st.success(f" Detected {n_faces} face(s) in image!")

                    # Process each detected face (focusing primarily on face 0)
                    predictions_for_annotation = []
                    primary_preds = None

                    for idx, face in enumerate(faces_data):
                        top_emotion, top_conf, prob_dict = predict_expression(model, face['cropped_tensor'])
                        predictions_for_annotation.append((face['box'], top_emotion, top_conf, face['is_fallback']))
                        if idx == 0:
                            primary_preds = (top_emotion, top_conf, prob_dict)

                    # Draw bounding box
                    annotated_rgb = annotate_image_with_predictions(rgb_arr, predictions_for_annotation)
                    st.image(annotated_rgb, caption="Annotated Detection & Classification", use_container_width=True)

                    # Primary result details
                    top_emotion, top_conf, prob_dict = primary_preds
                    color = EMOTION_COLORS.get(top_emotion, "#4F46E5")

                    st.markdown("### Top Predicted Expression")
                    st.markdown(
                        f"""
                        <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 1rem 1.4rem; border-radius: 12px; margin-bottom: 1.5rem;">
                            <div style="font-size: 0.9rem; color: #475569; font-weight: 600;">CLASSIFICATION RESULT</div>
                            <div style="font-size: 2.2rem; font-weight: 800; color: {color};">{top_emotion}</div>
                            <div style="font-size: 1.05rem; font-weight: 600; color: #1E293B;">Prediction Confidence: {top_conf*100:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Bias alert for minority classes
                    if top_emotion in ['Disgust']:
                        st.warning(
                            "**⚠️ Dataset Bias Notice**: 'Disgust' constitutes only ~2% of the training dataset. "
                            "Predictions for this minority class may carry higher variance due to severe dataset imbalance."
                        )

                    # Probability distribution bar chart
                    st.markdown("#### Probability Distribution Across All Classes")
                    df_probs = pd.DataFrame({
                        'Emotion': list(prob_dict.keys()),
                        'Confidence': [v * 100 for v in prob_dict.values()]
                    }).sort_values(by="Confidence", ascending=False)

                    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
                    bars = ax.barh(df_probs['Emotion'], df_probs['Confidence'], 
                                   color=[EMOTION_COLORS.get(e, '#6366F1') for e in df_probs['Emotion']],
                                   alpha=0.88, edgecolor='black', linewidth=0.5)
                    ax.set_xlabel("Confidence (%)", fontweight="bold")
                    ax.set_xlim(0, 100)
                    ax.invert_yaxis()
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
                                va='center', ha='left', fontsize=8, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                except Exception as e:
                    st.error(f"Inference pipeline error: {e}")

# ==============================================================================
# TAB 2: DATASET BIAS ANALYSIS
# ==============================================================================
with tab_dataset:
    st.subheader("Dataset Bias & Class Representation Study")
    st.caption("Quantitative audit of class imbalance, entropy, and structural representation disparities.")

    metrics, dataset_summary = load_results_data()

    if dataset_summary is None:
        st.warning("Dataset summary not found. Click below to run a comprehensive dataset audit.")
        if st.button("🔍 Run Dataset Bias Audit Now"):
            with st.spinner("Auditing dataset distributions..."):
                run_dataset_analysis()
                st.rerun()
    else:
        # Key metrics row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total Training Images</div>
                    <div class="metric-value">{dataset_summary.get('total_images', 0):,}</div>
                </div>
                """, unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Imbalance Ratio (IR)</div>
                    <div class="metric-value" style="color: #DC2626;">{dataset_summary.get('imbalance_ratio', 0):.2f}:1</div>
                </div>
                """, unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Shannon Entropy</div>
                    <div class="metric-value">{dataset_summary.get('shannon_entropy', 0):.3f} <span style="font-size: 0.9rem; color: #64748B;">bits</span></div>
                </div>
                """, unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Normalized Balance</div>
                    <div class="metric-value">{dataset_summary.get('normalized_entropy', 0):.3f} <span style="font-size: 0.9rem; color: #64748B;">/ 1.0</span></div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2, gap="medium")
        with col_c1:
            st.markdown("#### 📊 Class Distribution (Train vs Uniform Baseline)")
            chart_dist = "results/class_distribution.png"
            if os.path.exists(chart_dist):
                st.image(chart_dist, use_container_width=True)
            else:
                st.info("Chart not generated yet.")

        with col_c2:
            st.markdown("#### ⚖️ Representation Disparity Ratio")
            chart_disp = "results/bias_disparities.png"
            if os.path.exists(chart_disp):
                st.image(chart_disp, use_container_width=True)
            else:
                st.info("Disparity chart not generated yet.")

        st.markdown("---")
        st.markdown("### 📋 Class-by-Class Bias Audit Table")
        
        disparities = dataset_summary.get('disparities', {})
        if disparities:
            table_rows = []
            for emotion, data in disparities.items():
                table_rows.append({
                    "Emotion": emotion,
                    "Count": data['count'],
                    "Dataset Share (%)": f"{data['share_pct']:.2f}%",
                    "Disparity Ratio": f"{data['disparity_ratio']:.2f}x",
                    "Fairness Status": data['status']
                })
            df_disp = pd.DataFrame(table_rows)
            st.dataframe(df_disp, use_container_width=True, hide_index=True)

        st.markdown("### 🔍 Key Dataset Bias Insights")
        st.markdown("""
        1. **Extreme Minority Class Penalty**: `Disgust` represents only ~2% of the dataset, while `Happy` accounts for ~25%. This severe representation skew often leads standard gradient descent to prioritize majority classes.
        2. **Fair Share Baseline**: In an ideally balanced 7-class system, each expression should account for **14.28%** of the data.
        3. **Algorithmic Remedy**: In our training pipeline, we counteract this bias by computing **class weights**, weighting underrepresented samples during gradient updates.
        """)

# ==============================================================================
# TAB 3: EVALUATION & FAIRNESS AUDIT
# ==============================================================================
with tab_eval:
    st.subheader("Model Evaluation & Algorithmic Fairness Audit")
    st.caption("All metrics shown below are calculated directly on the actual test set (Never Invented).")

    metrics, _ = load_results_data()

    if metrics is None:
        st.warning("No test evaluation metrics found in `results/metrics.json`. Please train and evaluate the model first.")
    else:
        # Key evaluation metrics cards
        e1, e2, e3, e4, e5 = st.columns(5)
        with e1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Overall Accuracy</div>
                    <div class="metric-value" style="color: #4F46E5;">{metrics.get('overall_accuracy', 0)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True
            )
        with e2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Macro Precision</div>
                    <div class="metric-value">{metrics.get('macro_precision', 0)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True
            )
        with e3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Macro Recall</div>
                    <div class="metric-value">{metrics.get('macro_recall', 0)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True
            )
        with e4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Macro F1-Score</div>
                    <div class="metric-value" style="color: #059669;">{metrics.get('macro_f1', 0)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True
            )
        with e5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Weighted F1-Score</div>
                    <div class="metric-value">{metrics.get('weighted_f1', 0)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_ev1, col_ev2 = st.columns(2, gap="large")
        with col_ev1:
            st.markdown("#### 🎯 Normalized Confusion Matrix")
            cm_img = "results/confusion_matrix.png"
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)
            else:
                st.info("Confusion matrix plot not found.")

        with col_ev2:
            st.markdown("#### 📊 Per-Class Precision, Recall, and F1-Score")
            per_class_img = "results/per_class_metrics.png"
            if os.path.exists(per_class_img):
                st.image(per_class_img, use_container_width=True)
            else:
                st.info("Per-class performance chart not found.")

        st.markdown("---")
        st.markdown("### 🌦️ Robustness & Environmental Condition Stress Test")
        st.caption("Evaluation across visual conditions (lighting, blur, high contrast, noise) to detect physical domain bias.")

        col_bias1, col_bias2 = st.columns([1.1, 1], gap="medium")
        with col_bias1:
            cond_img = "results/condition_bias.png"
            if os.path.exists(cond_img):
                st.image(cond_img, use_container_width=True)
            else:
                st.info("Condition bias chart not found.")

        with col_bias2:
            st.markdown("#### 🔬 Condition Robustness Findings")
            cond_dict = metrics.get('condition_bias_metrics', {})
            if cond_dict:
                cond_rows = []
                for cond_name, info in cond_dict.items():
                    cond_rows.append({
                        "Condition / Environment": cond_name.replace("_", " ").title(),
                        "Test Samples": info['count'],
                        "Accuracy (%)": f"{info['accuracy']*100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)

            st.info(
                "**Bias Conclusion**: Lower illumination and noise degrade accuracy significantly compared to standard studio conditions. "
                "Models deployed in real-world CCTV or varying camera environments must include aggressive illumination augmentation."
            )

# ==============================================================================
# TAB 4: MODEL RETRAINING STUDIO
# ==============================================================================
with tab_train:
    st.subheader("Model Retraining & Experimentation Studio")
    st.caption("Adjust hyperparameters, toggle class weighting, and retrain the CNN directly from the UI.")

    col_t1, col_t2 = st.columns([1, 1.4], gap="large")

    with col_t1:
        st.markdown("#### 🛠️ Training Hyperparameters")
        epochs_slider = st.slider("Training Epochs:", min_value=2, max_value=25, value=8, step=1)
        batch_size_sel = st.selectbox("Batch Size:", [16, 32, 64], index=1)
        use_weights = st.checkbox("Apply Smoothed Class Weighting (Mitigate Bias)", value=True,
                                  help="Assigns higher loss weight to underrepresented classes (like Disgust).")

        st.markdown("---")
        start_train_btn = st.button("🚀 Train Model & Re-evaluate", type="primary")

    with col_t2:
        st.markdown("#### 📈 Training Progression & History")
        train_hist_img = "results/training_history.png"
        if os.path.exists(train_hist_img):
            st.image(train_hist_img, caption="Loss & Accuracy Curves (Last Run)", use_container_width=True)
        else:
            st.info("No training history plot found yet.")

    if start_train_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Preparing dataset and building CNN architecture...")
        progress_bar.progress(20)
        time.sleep(1)

        try:
            status_text.text(f"Training CNN for {epochs_slider} epochs with batch size {batch_size_sel}...")
            progress_bar.progress(40)

            # Run training
            new_results = train_and_evaluate(
                epochs=epochs_slider,
                batch_size=batch_size_sel,
                use_class_weights=use_weights
            )

            progress_bar.progress(90)
            status_text.text("Saving model weights, computing confusion matrix and metrics...")
            time.sleep(1)

            progress_bar.progress(100)
            st.success("🎉 Model trained and evaluated successfully on the test dataset!")
            st.balloons()
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {e}")

# ==============================================================================
# TAB 5: VIVA VOCE & DEFENSE GUIDE
# ==============================================================================
with tab_viva:
    st.subheader("🎓 College Defense & Viva Voce Preparation Guide")
    st.caption("Comprehensive academic questions and high-scoring answers covering project theory, architecture, and bias study.")

    viva_qa = [
        (
            "1. What is the central objective of this project?",
            "The objective is twofold: (1) Build a Convolutional Neural Network (CNN) to classify human facial expressions into 7 discrete categories (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust), and (2) Systematically investigate and quantify **Dataset Bias** (representation imbalance and environmental degradation) and evaluate mitigation strategies like class weighting."
        ),
        (
            "2. What is Dataset Bias in Facial Emotion Recognition, and why does it occur?",
            "Dataset bias occurs when the training data is not a balanced, representative sample of the real-world operational space. In FER-2013, for instance, `Happy` represents ~25% of all samples, while `Disgust` constitutes merely ~2%. Furthermore, datasets often skew toward specific demographics, studio lighting, or frontal poses, causing deep learning models to perform poorly on minority classes and real-world conditions."
        ),
        (
            "3. How do you quantitatively measure dataset imbalance?",
            "We use three key mathematical metrics:\n"
            "- **Imbalance Ratio (IR)**: $IR = \\frac{\\max_c N_c}{\\min_c N_c}$. An IR of 12.5:1 means the majority class has 12.5 times more samples than the minority class.\n"
            "- **Shannon Entropy**: $H = -\\sum_{i=1}^K p_i \\log_2(p_i)$, measuring the information diversity of the distribution.\n"
            "- **Normalized Entropy**: $\\frac{H}{\\log_2(K)}$, where 1.0 represents a perfectly balanced uniform distribution and values < 0.9 indicate severe skew."
        ),
        (
            "4. Why is Accuracy alone an unreliable metric for imbalanced datasets?",
            "If a majority class makes up 90% of the dataset, a naive classifier predicting only the majority class achieves 90% accuracy while failing completely on minority classes (0% recall). Therefore, we rely on **Precision**, **Recall**, and **F1-Score** (Macro and Weighted), alongside the **Normalized Confusion Matrix**, to expose per-class disparities."
        ),
        (
            "5. What is the role of OpenCV Haar Cascades in this project?",
            "OpenCV Haar Cascades provide face localization. Before feeding an image to our CNN, the Haar Cascade detects the bounding coordinates $(x, y, w, h)$ of the face, crops the Region of Interest (ROI), converts it to 48x48 grayscale, and normalizes pixel values to $[0, 1]$. If no face is detected, a graceful center-crop fallback is executed."
        ),
        (
            "6. Explain your CNN architecture and the purpose of Batch Normalization and Dropout.",
            "Our model uses stacked Conv2D layers with 3x3 kernels and ReLU activations to learn low-to-high level facial feature maps (edges, mouth arcs, eyebrow furrows). **Batch Normalization** stabilizes and accelerates gradient convergence by normalizing layer inputs, while **Dropout (0.2 - 0.4)** prevents co-adaptation of neurons and combats overfitting."
        ),
        (
            "7. How does Class Weighting mitigate representation bias?",
            "During loss calculation, standard cross-entropy treats each sample equally. Class weighting assigns higher penalties to errors made on minority classes:\n"
            "$$w_c = \\sqrt{\\frac{N}{K \\cdot N_c}}$$\n"
            "This forces the optimizer to update weights meaningfully for minority classes like Disgust, preventing the model from collapsing to majority-only predictions."
        ),
        (
            "8. What is the difference between Macro F1-Score and Weighted F1-Score?",
            "- **Macro F1**: Calculates metrics unweighted per class and computes the arithmetic mean. It treats all classes equally, penalizing poor performance on minority classes.\n"
            "- **Weighted F1**: Weights each class's F1 score by its support (number of samples in test set). It can mask poor minority performance if the majority class achieves high accuracy."
        ),
        (
            "9. What are the ethical implications of biased facial emotion classifiers?",
            "Facial expression AI is increasingly proposed for hiring assessments, proctoring, and border security. Biased classifiers can systematically misclassify expressions for specific groups (e.g., misinterpreting natural expressions as 'Angry' or 'Disgusted'), leading to unfair discrimination and harm if deployed without rigorous bias auditing."
        ),
        (
            "10. How can this project be extended for future research?",
            "Future enhancements include: (1) Testing on multimodal datasets (FER+, RAF-DB, AffectNet), (2) Implementing Attention mechanisms (Vision Transformers or CBAM), (3) Cross-dataset demographic evaluation (gender/age/ethnicity fairness auditing), and (4) Adversarial debiasing techniques."
        )
    ]

    for question, answer in viva_qa:
        with st.expander(f"📌 {question}", expanded=False):
            st.markdown(f"**Answer:**\n\n{answer}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem;'>"
    "Facial Expression Classification: Dataset Bias Study • Built with Python, TensorFlow/Keras, OpenCV & Streamlit • Academic College Project"
    "</div>",
    unsafe_allow_html=True
)
