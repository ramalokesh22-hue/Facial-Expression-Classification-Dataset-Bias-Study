"""
download_data.py
================
Dataset setup helper for Facial Expression Classification: Dataset Bias Study.

This script supports:
1. Preparing the FER-2013 dataset structure (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise).
2. Generating a starter dataset with realistic facial feature geometry, illumination variations,
   and representative class imbalance mirroring FER-2013 so that training, bias analysis,
   and model evaluation can be executed immediately offline without blockers.
3. Providing clear instructions to import full FER-2013 or FER+ datasets from Kaggle.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Proportions matching FER-2013 distribution (Happy is dominant, Disgust is severe minority)
FER2013_RATIOS = {
    'angry': 0.14,
    'disgust': 0.02,    # Severe minority class (~436 in FER2013)
    'fear': 0.14,
    'happy': 0.25,      # Dominant class (~7215 in FER2013)
    'neutral': 0.17,
    'sad': 0.17,
    'surprise': 0.11
}

def create_directory_structure(base_dir="data"):
    """Creates directory structure for train and test splits."""
    for split in ["train", "test"]:
        for emotion in CLASSES:
            path = os.path.join(base_dir, split, emotion)
            os.makedirs(path, exist_ok=True)
    print(f"[OK] Directory structure created at '{base_dir}/'")

def generate_face_image(emotion, condition="normal", seed=None):
    """
    Generates a realistic 48x48 grayscale face image with geometric expression features
    and specified lighting/condition attributes for bias study benchmarking.
    """
    if seed is not None:
        np.random.seed(seed)
        
    size = (48, 48)
    # Background tone
    bg_tone = np.random.randint(40, 80)
    img = Image.new("L", size, color=bg_tone)
    draw = ImageDraw.Draw(img)
    
    # Face oval (skin tone range 130 - 210)
    skin_tone = np.random.randint(140, 210)
    face_box = [7 + np.random.randint(-1, 2), 5, 41 + np.random.randint(-1, 2), 44]
    draw.ellipse(face_box, fill=skin_tone)
    
    # Eyes positions
    eye_y = 18 + np.random.randint(-1, 2)
    left_eye_x = 16
    right_eye_x = 32
    
    # Brows & Mouth vary by emotion
    if emotion == 'happy':
        # Smiling mouth (upward arc)
        draw.arc([16, 27, 32, 39], start=0, end=180, fill=40, width=2)
        # Relaxed curved eyebrows
        draw.line([13, eye_y - 4, 19, eye_y - 6], fill=50, width=1)
        draw.line([29, eye_y - 6, 35, eye_y - 4], fill=50, width=1)
        # Open smiling eyes
        draw.ellipse([left_eye_x - 3, eye_y - 1, left_eye_x + 3, eye_y + 2], fill=30)
        draw.ellipse([right_eye_x - 3, eye_y - 1, right_eye_x + 3, eye_y + 2], fill=30)
        
    elif emotion == 'sad':
        # Turned down mouth (downward arc)
        draw.arc([16, 33, 32, 43], start=180, end=360, fill=40, width=2)
        # Eyebrows angled up in center
        draw.line([13, eye_y - 3, 19, eye_y - 6], fill=50, width=1)
        draw.line([29, eye_y - 6, 35, eye_y - 3], fill=50, width=1)
        # Droopy eyes
        draw.ellipse([left_eye_x - 2, eye_y, left_eye_x + 2, eye_y + 2], fill=40)
        draw.ellipse([right_eye_x - 2, eye_y, right_eye_x + 2, eye_y + 2], fill=40)
        
    elif emotion == 'angry':
        # Tight straight or slight downturn mouth
        draw.line([17, 34, 31, 34], fill=30, width=2)
        # V-shaped furrowed eyebrows
        draw.line([13, eye_y - 6, 20, eye_y - 3], fill=30, width=2)
        draw.line([28, eye_y - 3, 35, eye_y - 6], fill=30, width=2)
        # Intense eyes
        draw.ellipse([left_eye_x - 2, eye_y - 1, left_eye_x + 2, eye_y + 2], fill=20)
        draw.ellipse([right_eye_x - 2, eye_y - 1, right_eye_x + 2, eye_y + 2], fill=20)
        
    elif emotion == 'surprise':
        # Round open mouth
        draw.ellipse([21, 30, 27, 39], fill=30)
        # High arched eyebrows
        draw.line([13, eye_y - 8, 19, eye_y - 9], fill=50, width=1)
        draw.line([29, eye_y - 9, 35, eye_y - 8], fill=50, width=1)
        # Wide open round eyes
        draw.ellipse([left_eye_x - 3, eye_y - 3, left_eye_x + 3, eye_y + 3], fill=30)
        draw.ellipse([right_eye_x - 3, eye_y - 3, right_eye_x + 3, eye_y + 3], fill=30)
        
    elif emotion == 'fear':
        # Open horizontal mouth
        draw.ellipse([17, 32, 31, 38], fill=40)
        # Raised, flattened eyebrows
        draw.line([14, eye_y - 7, 20, eye_y - 7], fill=40, width=1)
        draw.line([28, eye_y - 7, 34, eye_y - 7], fill=40, width=1)
        # Staring eyes
        draw.ellipse([left_eye_x - 3, eye_y - 2, left_eye_x + 3, eye_y + 2], fill=30)
        draw.ellipse([right_eye_x - 3, eye_y - 2, right_eye_x + 3, eye_y + 2], fill=30)
        
    elif emotion == 'disgust':
        # Wrinkled nose and grimace mouth
        draw.line([17, 35, 24, 32], fill=40, width=2)
        draw.line([24, 32, 31, 35], fill=40, width=2)
        # Nose wrinkle lines
        draw.line([22, 25, 26, 25], fill=50, width=1)
        # Narrowed eyes
        draw.line([left_eye_x - 3, eye_y, left_eye_x + 3, eye_y], fill=30, width=2)
        draw.line([right_eye_x - 3, eye_y, right_eye_x + 3, eye_y], fill=30, width=2)
        
    else: # neutral
        # Flat mouth line
        draw.line([17, 34, 31, 34], fill=50, width=2)
        # Horizontal standard brows
        draw.line([14, eye_y - 5, 20, eye_y - 5], fill=50, width=1)
        draw.line([28, eye_y - 5, 34, eye_y - 5], fill=50, width=1)
        # Normal eyes
        draw.ellipse([left_eye_x - 2, eye_y - 1, left_eye_x + 2, eye_y + 2], fill=40)
        draw.ellipse([right_eye_x - 2, eye_y - 1, right_eye_x + 2, eye_y + 2], fill=40)
        
    # Nose indicator
    draw.line([24, 23, 24, 27], fill=60, width=1)
    
    # Apply bias condition perturbation
    arr = np.array(img, dtype=np.float32)
    
    if condition == "low_light":
        arr = arr * 0.45  # Dark lighting
    elif condition == "high_contrast":
        arr = np.clip((arr - 128) * 1.5 + 128, 0, 255)
    elif condition == "noisy":
        noise = np.random.normal(0, 18, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
    elif condition == "blur":
        img_temp = Image.fromarray(np.uint8(arr)).filter(ImageFilter.GaussianBlur(radius=1.2))
        arr = np.array(img_temp, dtype=np.float32)
    else:
        # Subtle realistic camera noise
        noise = np.random.normal(0, 4, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        
    final_img = Image.fromarray(np.uint8(arr))
    return final_img

def generate_starter_dataset(base_dir="data", total_train=1400, total_test=350):
    """
    Generates a benchmark facial expression dataset with realistic class imbalance
    and varied lighting/noise conditions.
    """
    create_directory_structure(base_dir)
    print(f"Generating starter dataset ({total_train} train, {total_test} test images)...")
    
    conditions = ["normal", "low_light", "high_contrast", "noisy", "blur"]
    
    for split, count in [("train", total_train), ("test", total_test)]:
        for emotion in CLASSES:
            n_samples = max(10, int(count * FER2013_RATIOS[emotion]))
            out_folder = os.path.join(base_dir, split, emotion)
            
            for i in range(n_samples):
                # 70% normal, 30% challenging conditions to study condition bias
                cond = "normal" if np.random.rand() > 0.3 else np.random.choice(conditions)
                seed = i * 17 + len(emotion) * 31 + (1000 if split == "test" else 0)
                img = generate_face_image(emotion, condition=cond, seed=seed)
                file_name = f"{emotion}_{cond}_{i:04d}.png"
                img.save(os.path.join(out_folder, file_name))
                
    print(f"[OK] Starter dataset generated successfully in '{base_dir}/'")

def create_sample_test_images(samples_dir="sample_images"):
    """Creates a gallery of labeled sample faces for instant UI testing."""
    os.makedirs(samples_dir, exist_ok=True)
    for emotion in CLASSES:
        img = generate_face_image(emotion, condition="normal", seed=42)
        # Upscale slightly for nice preview while keeping 48x48 aspect ratio
        display_img = img.resize((192, 192), Image.Resampling.NEAREST)
        display_img.save(os.path.join(samples_dir, f"sample_{emotion}.png"))
    print(f"[OK] Sample test images saved in '{samples_dir}/'")

def print_fer2013_instructions():
    """Prints instructions on downloading the full FER-2013 dataset."""
    print("=" * 70)
    print("FER-2013 DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print("Option 1: Download from Kaggle directly")
    print("  URL: https://www.kaggle.com/datasets/msambare/fer2013")
    print("  1. Download the archive.zip file from Kaggle.")
    print("  2. Extract it. You will see 'train' and 'test' folders.")
    print("  3. Place 'train' and 'test' inside the 'data/' folder of this project:")
    print("     data/")
    print("       train/ (angry, disgust, fear, happy, neutral, sad, surprise)")
    print("       test/  (angry, disgust, fear, happy, neutral, sad, surprise)")
    print("")
    print("Option 2: Kaggle CLI (if configured)")
    print("  kaggle datasets download -d msambare/fer2013 --unzip -p ./data")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset setup for Facial Expression Classification Bias Study")
    parser.add_argument("--generate-starter", action="store_true", help="Generate starter dataset for instant testing")
    parser.add_argument("--info", action="store_true", help="Show FER-2013 download instructions")
    args = parser.parse_args()
    
    if args.info:
        print_fer2013_instructions()
    else:
        # Default action: create directory structure and starter dataset
        generate_starter_dataset()
        create_sample_test_images()
        print_fer2013_instructions()
