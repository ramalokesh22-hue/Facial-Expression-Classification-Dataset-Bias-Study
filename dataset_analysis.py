"""
dataset_analysis.py
===================
Analyzes the facial expression dataset to detect, measure, and visualize dataset bias.
Calculates class distributions, Imbalance Ratio, Shannon Entropy, and Representation Disparity.
Generates charts saved in `results/` and exports `results/dataset_summary.json`.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from utils import EMOTION_CLASSES, EMOTION_LOWER, calculate_dataset_bias_stats

def analyze_split(split_path):
    """Counts files and checks image validity in a split folder (train or test)."""
    counts = {}
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    if not os.path.exists(split_path):
        return {e: 0 for e in EMOTION_CLASSES}

    for emotion in EMOTION_CLASSES:
        counts[emotion] = 0
        # Check capitalized and lower-case variants
        candidates = [emotion, emotion.lower()]
        target_dir = None
        for c in candidates:
            p = os.path.join(split_path, c)
            if os.path.isdir(p):
                target_dir = p
                break
                
        if target_dir:
            file_list = [f for f in os.listdir(target_dir) 
                         if os.path.splitext(f)[1].lower() in valid_extensions]
            counts[emotion] = len(file_list)
            
    return counts

def analyze_image_conditions(base_dir="data"):
    """
    Categorizes images across condition attributes (e.g. normal, low-light, blur, contrast, noise)
    using filename tags or brightness/variance metrics.
    """
    conditions_count = {
        'normal': 0,
        'low_light': 0,
        'high_contrast': 0,
        'noisy': 0,
        'blur': 0
    }
    
    for split in ['train', 'test']:
        for emotion in EMOTION_CLASSES:
            p = os.path.join(base_dir, split, emotion.lower())
            if not os.path.isdir(p):
                p = os.path.join(base_dir, split, emotion)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    lower_f = f.lower()
                    if 'low_light' in lower_f or 'dark' in lower_f:
                        conditions_count['low_light'] += 1
                    elif 'contrast' in lower_f:
                        conditions_count['high_contrast'] += 1
                    elif 'noise' in lower_f:
                        conditions_count['noisy'] += 1
                    elif 'blur' in lower_f:
                        conditions_count['blur'] += 1
                    else:
                        conditions_count['normal'] += 1
                        
    return conditions_count

def generate_bias_visualizations(train_counts, test_counts, bias_stats, results_dir="results"):
    """Creates publication-quality visual charts for the dataset bias study."""
    os.makedirs(results_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Chart 1: Class Distribution Bar Chart (Train vs Test)
    emotions = EMOTION_CLASSES
    train_vals = [train_counts.get(e, 0) for e in emotions]
    test_vals = [test_counts.get(e, 0) for e in emotions]

    x = np.arange(len(emotions))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    rects1 = ax.bar(x - width/2, train_vals, width, label='Train Split', color='#4F46E5', alpha=0.9)
    rects2 = ax.bar(x + width/2, test_vals, width, label='Test Split', color='#06B6D4', alpha=0.9)

    # Reference uniform expectation line for train
    total_train = sum(train_vals)
    if total_train > 0:
        ideal_train = total_train / len(emotions)
        ax.axhline(ideal_train, color='#DC2626', linestyle='--', linewidth=1.5,
                   label=f'Uniform Balanced Baseline ({int(ideal_train)}/class)')

    ax.set_ylabel('Image Count', fontsize=12, fontweight='bold')
    ax.set_title('Facial Expression Dataset: Class Imbalance & Representation', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, fontsize=11, fontweight='medium')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

    # Label counts on bars
    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h}', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart1_path = os.path.join(results_dir, 'class_distribution.png')
    plt.savefig(chart1_path)
    plt.close()

    # Chart 2: Representation Disparity Ratio (Bias Metric)
    # Ratio > 1.0 = overrepresented, < 1.0 = underrepresented
    disparities = bias_stats.get('disparities', {})
    disp_values = [disparities.get(e, {}).get('disparity_ratio', 1.0) for e in emotions]
    colors = ['#EF4444' if v < 0.75 else ('#10B981' if v > 1.25 else '#3B82F6') for v in disp_values]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars = ax.barh(emotions, disp_values, color=colors, alpha=0.88, edgecolor='black', linewidth=0.5)
    ax.axvline(1.0, color='#111827', linestyle='-', linewidth=2, label='Fair Uniform Parity (1.0x)')
    ax.axvline(0.75, color='#DC2626', linestyle=':', linewidth=1.5, label='Underrepresentation Threshold (0.75x)')
    ax.axvline(1.25, color='#059669', linestyle=':', linewidth=1.5, label='Overrepresentation Threshold (1.25x)')

    ax.set_xlabel('Representation Disparity Ratio (Actual Share / Fair Share)', fontsize=11, fontweight='bold')
    ax.set_title('Dataset Bias Audit: Class Representation Disparity', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True)

    for bar, val in zip(bars, disp_values):
        ax.text(val + 0.03, bar.get_y() + bar.get_height()/2, f'{val:.2f}x',
                va='center', ha='left', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart2_path = os.path.join(results_dir, 'bias_disparities.png')
    plt.savefig(chart2_path)
    plt.close()

    print(f"[OK] Visual charts saved: '{chart1_path}' and '{chart2_path}'")

def run_dataset_analysis(base_dir="data", results_dir="results"):
    """Executes full dataset audit and exports summary metrics."""
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_counts = analyze_split(train_dir)
    test_counts = analyze_split(test_dir)
    condition_counts = analyze_image_conditions(base_dir)

    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())

    if total_train == 0 and total_test == 0:
        print(f"[WARNING] No dataset found in '{base_dir}'. Please run 'python download_data.py --generate-starter' first.")
        return None

    # Calculate Bias Stats for training set
    bias_stats = calculate_dataset_bias_stats(train_counts)
    bias_stats['test_counts'] = test_counts
    bias_stats['condition_counts'] = condition_counts

    # Generate charts
    generate_bias_visualizations(train_counts, test_counts, bias_stats, results_dir)

    # Export to JSON
    summary_path = os.path.join(results_dir, "dataset_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(bias_stats, f, indent=4)
        
    print("=" * 60)
    print("DATASET BIAS AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Training Images: {bias_stats['total_images']}")
    print(f"Total Testing Images:  {sum(test_counts.values())}")
    print(f"Number of Classes:     {bias_stats['num_classes']}")
    print(f"Imbalance Ratio (IR):  {bias_stats['imbalance_ratio']:.2f}:1")
    print(f"Normalized Entropy:    {bias_stats['normalized_entropy']:.4f} (1.0 = perfectly balanced)")
    print("-" * 60)
    print("Class Representation Disparities:")
    for cls, info in bias_stats['disparities'].items():
        print(f"  {cls:10s} -> Count: {info['count']:5d} ({info['share_pct']:5.1f}%) | "
              f"Disparity: {info['disparity_ratio']:.2f}x [{info['status']}]")
    print("=" * 60)
    print(f"[OK] Summary exported to: {summary_path}")
    return bias_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset for representation bias and class imbalance.")
    parser.add_argument("--data-dir", default="data", help="Path to base dataset directory containing train/ and test/")
    parser.add_argument("--results-dir", default="results", help="Path to save output charts and metrics")
    args = parser.parse_args()

    run_dataset_analysis(args.data_dir, args.results_dir)
