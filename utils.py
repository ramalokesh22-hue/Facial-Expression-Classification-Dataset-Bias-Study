"""
utils.py
========
Image processing, face detection, prediction, and bias metrics utilities.
Supports OpenCV Haar-Cascade face detection, preprocessing, and model inference.
"""

import os
import cv2
import numpy as np
from PIL import Image

EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_LOWER = [e.lower() for e in EMOTION_CLASSES]

# Standard vibrant color mapping for emotions
EMOTION_COLORS = {
    'Angry': '#EF4444',     # Crimson
    'Disgust': '#10B981',   # Emerald
    'Fear': '#8B5CF6',      # Purple
    'Happy': '#F59E0B',     # Amber
    'Neutral': '#6B7280',   # Slate
    'Sad': '#3B82F6',       # Blue
    'Surprise': '#EC4899'   # Pink
}

def get_face_cascade():
    """Loads OpenCV's pre-trained Haar Cascade for frontal face detection."""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade XML not found at {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)

def detect_and_crop_faces(image_input, target_size=(48, 48)):
    """
    Detects faces in an image (either file path, PIL Image, or numpy array).
    Returns:
        faces_data: List of dicts containing:
            'box': (x, y, w, h)
            'cropped_tensor': normalized (1, 48, 48, 1) float32 array
            'cropped_display': 48x48 uint8 image for display
        original_rgb: Original image as RGB numpy array
        detected_count: Number of faces detected
    """
    # Convert input to RGB numpy array
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image file not found: {image_input}")
        bgr = cv2.imread(image_input)
        if bgr is None:
            raise ValueError(f"Could not decode image at {image_input}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            rgb = image_input.copy()
    else:
        raise TypeError("image_input must be a file path, PIL Image, or numpy ndarray")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = get_face_cascade()
    detected = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30)
    )

    faces_data = []

    if len(detected) == 0:
        # Fallback: Center-crop or whole image resize if no face is detected by Haar Cascade
        h, w = gray.shape
        crop_size = min(h, w)
        start_y = (h - crop_size) // 2
        start_x = (w - crop_size) // 2
        crop_gray = gray[start_y:start_y+crop_size, start_x:start_x+crop_size]
        resized = cv2.resize(crop_gray, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        tensor = np.expand_dims(normalized, axis=(0, -1)) # Shape: (1, 48, 48, 1)

        faces_data.append({
            'box': (start_x, start_y, crop_size, crop_size),
            'cropped_tensor': tensor,
            'cropped_display': resized,
            'is_fallback': True
        })
    else:
        for (x, y, w, h) in detected:
            face_roi = gray[y:y+h, x:x+w]
            resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
            normalized = resized.astype("float32") / 255.0
            tensor = np.expand_dims(normalized, axis=(0, -1))

            faces_data.append({
                'box': (x, y, w, h),
                'cropped_tensor': tensor,
                'cropped_display': resized,
                'is_fallback': False
            })

    return faces_data, rgb, len(detected)

def annotate_image_with_predictions(original_rgb, faces_predictions):
    """
    Draws bounding boxes and predicted emotion tags onto the original RGB image.
    faces_predictions is a list of tuples: (box, top_emotion, confidence, is_fallback)
    """
    annotated = original_rgb.copy()
    h_img, w_img = annotated.shape[:2]
    line_thickness = max(2, int(min(h_img, w_img) / 200))
    font_scale = max(0.5, min(h_img, w_img) / 450)

    for (x, y, w, h), emotion, conf, is_fallback in faces_predictions:
        # BGR color for OpenCV drawing
        color_hex = EMOTION_COLORS.get(emotion, "#6B7280").lstrip("#")
        r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        draw_color = (r, g, b)

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), draw_color, line_thickness)

        # Label badge
        fallback_tag = " (Full)" if is_fallback else ""
        label = f"{emotion}: {conf*100:.1f}%{fallback_tag}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        
        # Label background
        y_text = max(y - 8, label_h + 8)
        cv2.rectangle(
            annotated,
            (x, y_text - label_h - 4),
            (x + label_w + 8, y_text + baseline),
            draw_color,
            -1
        )
        # Label text (white)
        cv2.putText(
            annotated,
            label,
            (x + 4, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, int(line_thickness / 2)),
            cv2.LINE_AA
        )

    return annotated

def predict_expression(model, cropped_tensor):
    """
    Performs inference with the Keras model.
    Returns:
        top_emotion: str (e.g. 'Happy')
        top_confidence: float (e.g. 0.94)
        prob_dict: dict mapping each class name to its probability
    """
    preds = model.predict(cropped_tensor, verbose=0)[0]
    prob_dict = {EMOTION_CLASSES[i]: float(preds[i]) for i in range(len(EMOTION_CLASSES))}
    top_idx = int(np.argmax(preds))
    top_emotion = EMOTION_CLASSES[top_idx]
    top_confidence = float(preds[top_idx])
    return top_emotion, top_confidence, prob_dict

def calculate_dataset_bias_stats(class_counts):
    """
    Calculates quantitative dataset bias & representation metrics:
    - Imbalance Ratio (IR): max_count / min_count
    - Gini Impurity: sum(p_i * (1 - p_i))
    - Shannon Entropy (bits): -sum(p_i * log2(p_i))
    - Normalized Entropy: Entropy / log2(K)
    - Disparity table: comparison of each class share vs ideal balanced share (1/K)
    """
    total = sum(class_counts.values())
    if total == 0:
        return {}
    
    k = len(class_counts)
    ideal_pct = 100.0 / k
    probs = [cnt / total for cnt in class_counts.values()]
    
    # Imbalance Ratio
    max_c = max(class_counts.values())
    min_c = max(1, min(class_counts.values()))
    imbalance_ratio = max_c / min_c

    # Shannon Entropy
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(k)
    normalized_entropy = entropy / max_entropy

    # Disparity analysis
    disparities = {}
    for cls, cnt in class_counts.items():
        share_pct = (cnt / total) * 100.0
        disparity_ratio = share_pct / ideal_pct
        status = "Overrepresented" if disparity_ratio > 1.25 else ("Underrepresented" if disparity_ratio < 0.75 else "Balanced")
        disparities[cls] = {
            'count': cnt,
            'share_pct': share_pct,
            'disparity_ratio': disparity_ratio,
            'status': status
        }

    return {
        'total_images': total,
        'num_classes': k,
        'imbalance_ratio': imbalance_ratio,
        'shannon_entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'ideal_class_share_pct': ideal_pct,
        'disparities': disparities
    }
