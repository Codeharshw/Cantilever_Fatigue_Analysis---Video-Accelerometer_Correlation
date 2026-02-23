"""
CANTILEVER FATIGUE ANALYSIS - FEATURE EXTRACTION FROM MP4 VIDEOS
===================================================================

This version extracts frames directly from MP4 files and processes them
in real-time without saving intermediate PNG files.

Purpose: Extract temporal features from video microscopy data for correlation 
         with accelerometer measurements in fatigue experiments.

Features Extracted:
1. Rate of Change (ROC) - First derivative of intensity
2. Sudden Shifts - Second derivative and anomaly detection
3. Degradation Slope - Linear regression over sliding windows
4. Vector representation for multi-sensor correlation

Author: Research-grade implementation
Date: 2026-02-13
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d
import json
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE REQUIRED: Update filenames and time bounds to match your  │
# │  MP4 files.                                                            │
# │                                                                        │
# │  Format: (filename, period_start_hour, period_end_hour,               │
# │           skip_start_seconds, is_part2)                               │
# │                                                                        │
# │  Timeline logic:                                                       │
# │    is_part2=False → experiment_time = period_start + elapsed          │
# │    is_part2=True  → experiment_time = period_end - duration + elapsed │
# │                                                                        │
# │  Use is_part2=True when a video covers the END of a period (e.g. the  │
# │  camera was restarted mid-period and you only have the tail end).      │
# │  The gap in the middle is inferred automatically.                      │
# │                                                                        │
# │  skip_start_seconds: seconds to skip at the start of a video          │
# │  (e.g. 220 to skip 3.5 min of blurry warmup footage).                │
# │                                                                        │
# │  If your recordings are continuous (no mid-period restarts), simply   │
# │  set is_part2=False and skip_start_seconds=0 for all entries.         │
# └─────────────────────────────────────────────────────────────────────────┘
SEGMENT_CONFIG = [
    # 24-hour period (0-24h)
    ("24_hour_data_1.mp4", 0, 24, 220, False),   # Part 1: starts at 0h, skip blur
    ("24_hour_data_2.mp4", 0, 24, 0, True),      # Part 2: ends at 24h
    
    # 48-hour period (24-48h)  
    ("cantilever_48H_part1.mp4", 24, 48, 0, False),  # Part 1: starts at 24h
    ("cantilever_48H_part2.mp4", 24, 48, 0, True),   # Part 2: ends at 48h
    
    # 72-hour period (48-72h)
    ("72_hour_data_part1.mp4", 48, 72, 0, False),  # Part 1: starts at 48h
    ("72_hour_data_part2.mp4", 48, 72, 0, True),   # Part 2: ends at 72h
]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE REQUIRED: Update CSV filenames and period bounds to match │
# │  your accelerometer data files. Each entry maps one CSV to the         │
# │  24-hour period it covers, and lists which video files belong to it.  │
# │  This mapping is written into analysis_metadata.json and consumed by  │
# │  accelerometer_video_correlation.py.                                  │
# └─────────────────────────────────────────────────────────────────────────┘
CSV_VIDEO_MAPPING = {
    'cantilever_run_2_24_hour.csv': {
        'period_start': 0,
        'period_end': 24,
        'videos': ['24_hour_data_1.mp4', '24_hour_data_2.mp4']
    },
    'cantilever_run_48H.csv': {
        'period_start': 24,
        'period_end': 48,
        'videos': ['cantilever_48H_part1.mp4', 'cantilever_48H_part2.mp4']
    },
    'cantilever_run_72_hour_data.csv': {
        'period_start': 48,
        'period_end': 72,
        'videos': ['72_hour_data_part1.mp4', '72_hour_data_part2.mp4']
    }
}

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE REQUIRED: Set FPS to match your camera's actual frame     │
# │  rate. This is used to convert frame numbers to real timestamps.       │
# └─────────────────────────────────────────────────────────────────────────┘
FPS = 10  # Frames per second
FRAMES_PER_HOUR = FPS * 3600  # 36,000 frames per hour

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE (optional): Set VIDEO_DIR if your MP4 files are in a      │
# │  different folder from this script.                                    │
# │  TEMPLATE_FILENAME: name of the saved indent template image. Delete    │
# │  this file to force re-selection of the template region on next run.  │
# └─────────────────────────────────────────────────────────────────────────┘
VIDEO_DIR = "."  # Current directory (same folder as script)
TEMPLATE_FILENAME = "indent_template_1.png"

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE (optional): Tune feature extraction sensitivity.          │
# │                                                                        │
# │  shift_threshold: z-score cutoff for flagging sudden intensity shifts. │
# │    Higher = fewer shifts detected. Lower = more (noisier) detections.  │
# │  slope_window: number of consecutive samples used per local regression.│
# │    Larger = smoother slope estimate, less sensitive to short events.   │
# │  smoothing_sigma: Gaussian smoothing strength on the intensity signal. │
# │    Higher = smoother, fewer noise artifacts. Lower = more detail.     │
# └─────────────────────────────────────────────────────────────────────────┘
FEATURE_CONFIG = {
    'roc_window': 5,          # Window for rate of change calculation
    'shift_threshold': 3.0,    # Standard deviations for shift detection
    'slope_window': 20,        # Window for degradation slope (frames)
    'smoothing_sigma': 2.0,    # Gaussian smoothing parameter
    'anomaly_sensitivity': 2.5 # Sensitivity for anomaly detection
}

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE (optional): Controls how densely the video is sampled.   │
# │                                                                        │
# │  SAMPLING_INTERVAL_MINUTES = 2  →  one frame every 2 minutes          │
# │    = 720 samples over 24 hours instead of 864,000 raw frames          │
# │                                                                        │
# │  Increase to speed up processing. Decrease for finer time resolution. │
# │  Set to 0 to process every single frame (very slow).                  │
# │                                                                        │
# │  Should match window_minutes in accelerometer_video_correlation.py    │
# │  so each video sample has a corresponding accelerometer window.       │
# └─────────────────────────────────────────────────────────────────────────┘
SAMPLING_INTERVAL_MINUTES = 2  # Take one sample every N minutes (2 = one frame per 2 minutes)
# This processes the ENTIRE video duration but samples at intervals
# Example: 2 minutes = 720 samples over 24 hours (instead of 864,000 frames)
# 1 minute = 1,440 samples over 24 hours
# 5 minutes = 288 samples over 24 hours
# 0 or 1/60 = process every frame (full temporal resolution)

# For 24-hour video at 10fps with 2-minute sampling:
# - Total frames: 864,000
# - Sampled frames: 720 (one every 2 minutes)
# - Reduction: 1,200× fewer frames
# - Temporal coverage: Complete 24 hours
# - Time resolution: 2-minute intervals

# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_video_info(video_path):
    """Get video properties."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration_seconds': duration,
        'duration_hours': duration / 3600
    }

def normalize_brightness(img, ref_stats):
    """Normalize brightness to reference statistics."""
    if len(img.shape) == 3:
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        src_gray = img
        
    m_src, s_src = cv2.meanStdDev(src_gray)
    m_ref, s_ref = ref_stats
    
    if s_src[0][0] < 1e-5: 
        return img
    
    gain = s_ref[0][0] / s_src[0][0]
    offset = m_ref[0][0] - (m_src[0][0] * gain)
    
    normalized = cv2.convertScaleAbs(img, alpha=gain, beta=offset)
    return normalized

def load_or_create_template(img_gray):
    """Load or create indent template."""
    if os.path.exists(TEMPLATE_FILENAME):
        template = cv2.imread(TEMPLATE_FILENAME, 0)
        if template is not None:
            if template.shape[0] > img_gray.shape[0] // 2:
                template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
            print(f"✓ Loaded template from {TEMPLATE_FILENAME}")
            return template
    
    print(f"\n⚠ '{TEMPLATE_FILENAME}' not found!")
    print("Please select an indent region from the first frame...")
    
    try:
        r = cv2.selectROI("Select Indent Template", img_gray, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Indent Template")
        x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])
    except Exception as e:
        print(f"⚠ UI Selection failed: {e}")
        # Auto-crop center as fallback
        img_h, img_w = img_gray.shape
        w, h = 50, 50
        x = max(0, img_w // 2 - w // 2)
        y = max(0, img_h // 2 - h // 2)
    
    if w == 0 or h == 0:
        print("⚠ Selection invalid. Auto-cropping center.")
        img_h, img_w = img_gray.shape
        w, h = 50, 50
        x = max(0, img_w // 2 - w // 2)
        y = max(0, img_h // 2 - h // 2)
    
    template = img_gray[y:y+h, x:x+w]
    cv2.imwrite(TEMPLATE_FILENAME, template)
    print(f"✓ Saved template to {TEMPLATE_FILENAME}")
    return template

def align_image_ecc(target, reference):
    """ECC-based image alignment."""
    if len(reference.shape) == 3: 
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if len(target.shape) == 3: 
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    
    try:
        (cc, warp_matrix) = cv2.findTransformECC(
            reference, target, warp_matrix, warp_mode, criteria
        )
        h, w = reference.shape
        aligned_image = cv2.warpAffine(
            target, warp_matrix, (w, h), 
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
        return aligned_image
    except Exception:
        return None

def get_tracking_template(img, x, y, size=40):
    """Extract tracking template around point."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    h, w = gray.shape
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)
    return gray[y1:y2, x1:x2]

def track_template_local(img, start_x, start_y, tracking_template, search_window=80):
    """Track template in local search window."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    h, w = gray.shape
    t_h, t_w = tracking_template.shape
    
    x1 = max(0, start_x - search_window)
    y1 = max(0, start_y - search_window)
    x2 = min(w, start_x + search_window)
    y2 = min(h, start_y + search_window)
    
    roi = gray[y1:y2, x1:x2]
    
    if roi.shape[0] < t_h or roi.shape[1] < t_w:
        return start_x, start_y, 0.0
    
    res = cv2.matchTemplate(roi, tracking_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    match_center_x = x1 + max_loc[0] + t_w // 2
    match_center_y = y1 + max_loc[1] + t_h // 2
    
    return match_center_x, match_center_y, max_val

def get_indent_intensity(img, cx, cy, radius=8):
    """Measure average intensity in circular region."""
    if len(img.shape) == 3: 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(img)
    cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
    mean_val = cv2.mean(img, mask=mask)[0]
    return mean_val

def enhance_image_for_display(img):
    """Enhance image contrast and sharpness for better visibility during selection."""
    # Work with grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = img
        is_color = False
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply unsharp mask for better edge definition
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    
    # Convert back to color if needed
    if is_color:
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return sharpened

# Manual selection function
last_click = None

def click_selection_callback(event, x, y, flags, param):
    global last_click
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click = (x, y)

def select_indents_manual(img, template, title_prefix=""):
    """Manual indent selection with template snapping."""
    global last_click
    last_click = None
    
    print(f"\n{'='*60}")
    print(f"MANUAL INDENT SELECTION: {title_prefix}")
    print("CONTROLS:")
    print("  - CLICK near indents to select")
    print("  - Press 'C' to CONFIRM selections")
    print("  - Press 'D' to DELETE last selection")
    print("  - Press '+' to ZOOM IN (2x)")
    print("  - Press '-' to ZOOM OUT")
    print("  - Press 'R' to RESET zoom")
    print(f"{'='*60}")
    
    selected_indents = []
    window_name = f"Select Indents: {title_prefix}"
    
    # Create window with proper settings for clarity
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    
    # Get screen resolution and image size
    img_height, img_width = img.shape[:2]
    
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │  USER CHANGE (optional): Adjust screen_width / screen_height to    │
    # │  match your monitor resolution for a better display window size.   │
    # └─────────────────────────────────────────────────────────────────────┘
    screen_width = 1920  # Adjust if you have different resolution
    screen_height = 1080
    
    scale_w = screen_width * 0.8 / img_width  # Use 80% of screen width
    scale_h = screen_height * 0.8 / img_height
    scale = min(scale_w, scale_h, 2.0)  # Max 2x zoom
    
    display_width = int(img_width * scale)
    display_height = int(img_height * scale)
    
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, click_selection_callback)
    
    print(f"  Window size: {display_width}×{display_height} (scale: {scale:.2f}x)")
    print(f"  Image size: {img_width}×{img_height}")
    
    th, tw = template.shape[:2]
    search_radius = 30
    
    # Zoom state
    zoom_level = 1.0
    zoom_center_x = img_width // 2
    zoom_center_y = img_height // 2
    
    # Create enhanced version for display
    img_enhanced = enhance_image_for_display(img)
    
    while True:
        # Apply zoom if needed
        if zoom_level > 1.0:
            # Calculate zoom window
            zoom_w = int(img_width / zoom_level)
            zoom_h = int(img_height / zoom_level)
            
            x1 = max(0, zoom_center_x - zoom_w // 2)
            y1 = max(0, zoom_center_y - zoom_h // 2)
            x2 = min(img_width, x1 + zoom_w)
            y2 = min(img_height, y1 + zoom_h)
            
            # Adjust if at boundary
            if x2 - x1 < zoom_w:
                x1 = max(0, x2 - zoom_w)
            if y2 - y1 < zoom_h:
                y1 = max(0, y2 - zoom_h)
            
            display_img = img_enhanced[y1:y2, x1:x2].copy()
            
            # Adjust indent coordinates for zoomed view
            display_indents = []
            for sx, sy in selected_indents:
                if x1 <= sx < x2 and y1 <= sy < y2:
                    display_indents.append((sx - x1, sy - y1))
        else:
            display_img = img_enhanced.copy()
            display_indents = selected_indents
        
        # Draw selected indents with clear, visible markers
        for i, (sx, sy) in enumerate(display_indents):
            # Outer circle (larger, more visible)
            cv2.circle(display_img, (sx, sy), 25, (0, 0, 255), 3)
            # Inner dot
            cv2.circle(display_img, (sx, sy), 3, (0, 0, 255), -1)
            # Label with background for better visibility
            label = f"#{i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle
            bg_x1 = sx + 30
            bg_y1 = sy - text_height - 5
            bg_x2 = sx + 30 + text_width + 10
            bg_y2 = sy + 5
            cv2.rectangle(display_img, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                         (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(display_img, label, (sx+35, sy), 
                       font, font_scale, (0, 0, 0), font_thickness)
            
        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(20) & 0xFF
        
        if last_click:
            mx, my = last_click
            last_click = None
            
            h, w = img.shape[:2]
            x1 = max(0, mx - search_radius)
            y1 = max(0, my - search_radius)
            x2 = min(w, mx + search_radius)
            y2 = min(h, my + search_radius)
            
            roi = img[y1:y2, x1:x2]
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                roi_gray = roi
                
            if roi_gray.shape[0] >= th and roi_gray.shape[1] >= tw:
                try:
                    res = cv2.matchTemplate(roi_gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    
                    center_x = x1 + max_loc[0] + tw // 2
                    center_y = y1 + max_loc[1] + th // 2
                    
                    is_duplicate = any(
                        np.linalg.norm(np.array(e) - np.array((center_x, center_y))) < 10
                        for e in selected_indents
                    )
                    
                    if not is_duplicate:
                        selected_indents.append((center_x, center_y))
                        print(f"   ✓ Indent {len(selected_indents)} at ({center_x}, {center_y})")
                        
                except Exception as e:
                    print(f"   ⚠ Error: {e}")

        if key == ord('d') and selected_indents:
            removed = selected_indents.pop()
            print(f"   ✗ Removed indent at {removed}")
        
        elif key == ord('+') or key == ord('='):
            zoom_level = min(zoom_level * 2.0, 8.0)
            print(f"   Zoom: {zoom_level:.1f}x")
        
        elif key == ord('-') or key == ord('_'):
            zoom_level = max(zoom_level / 2.0, 1.0)
            print(f"   Zoom: {zoom_level:.1f}x")
        
        elif key == ord('r') or key == ord('R'):
            zoom_level = 1.0
            zoom_center_x = img_width // 2
            zoom_center_y = img_height // 2
            print(f"   Zoom reset")
                
        elif key in [ord('c'), ord('C')]:
            if selected_indents:
                cv2.destroyWindow(window_name)
                return selected_indents
            print("   ⚠ Select at least one indent")
                
        elif key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return []

# ==========================================
# FEATURE EXTRACTION CLASSES
# ==========================================

class FeatureExtractor:
    """Extracts temporal features from intensity time series."""
    
    def __init__(self, config=FEATURE_CONFIG):
        self.config = config
        
    def smooth_signal(self, signal_data, sigma=None):
        """Apply Gaussian smoothing."""
        if sigma is None:
            sigma = self.config['smoothing_sigma']
        return gaussian_filter1d(signal_data, sigma=sigma)
    
    def compute_rate_of_change(self, times, intensities):
        """Compute instantaneous rate of change (ROC)."""
        smoothed = self.smooth_signal(intensities)
        roc_values = np.gradient(smoothed, times)
        roc_values = self.smooth_signal(roc_values, sigma=1.0)
        return times, roc_values
    
    def detect_sudden_shifts(self, times, intensities):
        """Detect sudden shifts using second derivative and z-score."""
        smoothed = self.smooth_signal(intensities)
        first_deriv = np.gradient(smoothed, times)
        second_deriv = np.gradient(first_deriv, times)
        
        mean_2nd = np.mean(second_deriv)
        std_2nd = np.std(second_deriv)
        
        if std_2nd < 1e-6:
            return np.array([]), np.array([]), np.array([])
        
        z_scores = np.abs((second_deriv - mean_2nd) / std_2nd)
        threshold = self.config['shift_threshold']
        shift_indices = np.where(z_scores > threshold)[0]
        
        # Filter close events
        if len(shift_indices) > 0:
            filtered_indices = []
            i = 0
            while i < len(shift_indices):
                idx = shift_indices[i]
                window_start = max(0, i - 2)
                window_end = min(len(shift_indices), i + 3)
                window_indices = shift_indices[window_start:window_end]
                
                local_max_idx = window_indices[
                    np.argmax(z_scores[window_indices])
                ]
                
                if local_max_idx not in filtered_indices:
                    filtered_indices.append(local_max_idx)
                
                i += 1
                while i < len(shift_indices) and shift_indices[i] <= local_max_idx:
                    i += 1
            
            shift_indices = np.array(filtered_indices)
        
        shift_times = times[shift_indices]
        shift_magnitudes = second_deriv[shift_indices]
        
        return shift_times, shift_magnitudes, shift_indices
    
    def compute_degradation_slope(self, times, intensities):
        """Compute degradation slope using sliding window linear regression."""
        window = self.config['slope_window']
        
        if len(times) < window:
            return np.array([times[len(times)//2]]), \
                   np.array([0.0]), \
                   np.array([0.0])
        
        smoothed = self.smooth_signal(intensities)
        
        slope_times = []
        slope_values = []
        r_squared_values = []
        
        for i in range(len(times) - window + 1):
            t_window = times[i:i+window]
            i_window = smoothed[i:i+window]
            
            slope, intercept, r_value, p_value, std_err = linregress(
                t_window, i_window
            )
            
            center_time = np.mean(t_window)
            slope_times.append(center_time)
            slope_values.append(slope)
            r_squared_values.append(r_value**2)
        
        return (np.array(slope_times), 
                np.array(slope_values), 
                np.array(r_squared_values))
    
    def extract_all_features(self, times, intensities):
        """Extract all features and return as dictionary."""
        times = np.array(times)
        intensities = np.array(intensities)
        
        roc_times, roc_values = self.compute_rate_of_change(times, intensities)
        shift_times, shift_mags, shift_idx = self.detect_sudden_shifts(times, intensities)
        slope_times, slope_values, r_squared = self.compute_degradation_slope(times, intensities)
        
        stats = {
            'mean_intensity': np.mean(intensities),
            'std_intensity': np.std(intensities),
            'min_intensity': np.min(intensities),
            'max_intensity': np.max(intensities),
            'total_change': intensities[-1] - intensities[0],
            'mean_roc': np.mean(roc_values),
            'std_roc': np.std(roc_values),
            'num_shifts': len(shift_times),
            'mean_slope': np.mean(slope_values),
            'total_duration': times[-1] - times[0]
        }
        
        return {
            'times': times,
            'intensities': intensities,
            'roc_times': roc_times,
            'roc_values': roc_values,
            'shift_times': shift_times,
            'shift_magnitudes': shift_mags,
            'shift_indices': shift_idx,
            'slope_times': slope_times,
            'slope_values': slope_values,
            'r_squared': r_squared,
            'statistics': stats
        }

class VectorEncoder:
    """Converts temporal features into fixed-length vectors for correlation."""
    
    def __init__(self, n_bins=100):
        self.n_bins = n_bins
    
    def time_binned_features(self, features, time_range=None):
        """Create time-binned feature vectors."""
        times = features['times']
        
        if time_range is None:
            time_range = (times[0], times[-1])
        
        bins = np.linspace(time_range[0], time_range[1], self.n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        intensity_binned, _ = np.histogram(times, bins=bins, weights=features['intensities'])
        counts, _ = np.histogram(times, bins=bins)
        intensity_binned = np.divide(intensity_binned, counts, out=np.zeros_like(intensity_binned), where=counts > 0)
        
        roc_binned, _ = np.histogram(features['roc_times'], bins=bins, weights=features['roc_values'])
        roc_counts, _ = np.histogram(features['roc_times'], bins=bins)
        roc_binned = np.divide(roc_binned, roc_counts, out=np.zeros_like(roc_binned), where=roc_counts > 0)
        
        slope_binned, _ = np.histogram(features['slope_times'], bins=bins, weights=features['slope_values'])
        slope_counts, _ = np.histogram(features['slope_times'], bins=bins)
        slope_binned = np.divide(slope_binned, slope_counts, out=np.zeros_like(slope_binned), where=slope_counts > 0)
        
        shift_binned = np.zeros(self.n_bins)
        if len(features['shift_times']) > 0:
            shift_bin_indices = np.digitize(features['shift_times'], bins) - 1
            shift_bin_indices = np.clip(shift_bin_indices, 0, self.n_bins - 1)
            for idx in shift_bin_indices:
                shift_binned[idx] = 1
        
        feature_vector = np.column_stack([intensity_binned, roc_binned, slope_binned, shift_binned])
        
        return feature_vector, bin_centers
    
    def statistical_summary_vector(self, features):
        """Create statistical summary vector (fixed length)."""
        stats = features['statistics']
        
        intensity_percentiles = np.percentile(features['intensities'], [0, 25, 50, 75, 100])
        roc_percentiles = np.percentile(features['roc_values'], [0, 25, 50, 75, 100])
        slope_percentiles = np.percentile(features['slope_values'], [0, 25, 50, 75, 100])
        
        total_time = stats['total_duration']
        shift_rate = stats['num_shifts'] / total_time if total_time > 0 else 0
        
        vector = np.array([
            stats['mean_intensity'], stats['std_intensity'], *intensity_percentiles,
            stats['mean_roc'], stats['std_roc'], *roc_percentiles,
            stats['mean_slope'], *slope_percentiles,
            stats['num_shifts'], shift_rate, stats['total_change'], total_time
        ])
        
        labels = [
            'mean_intensity', 'std_intensity',
            'intensity_min', 'intensity_25', 'intensity_50', 'intensity_75', 'intensity_max',
            'mean_roc', 'std_roc',
            'roc_min', 'roc_25', 'roc_50', 'roc_75', 'roc_max',
            'mean_slope',
            'slope_min', 'slope_25', 'slope_50', 'slope_75', 'slope_max',
            'num_shifts', 'shift_rate', 'total_change', 'total_duration'
        ]
        
        return vector, labels

# ==========================================
# VISUALIZATION
# ==========================================

def plot_comprehensive_analysis(indent_id, features, output_path=None):
    """Create comprehensive visualization of all features."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    times = features['times']
    intensities = features['intensities']
    
    # 1. Raw Intensity Time Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, intensities, 'b-', linewidth=1, alpha=0.7, label='Raw')
    smoothed = gaussian_filter1d(intensities, sigma=2.0)
    ax1.plot(times, smoothed, 'r-', linewidth=2, label='Smoothed')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'Indent {indent_id}: Intensity Time Series', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Rate of Change
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(features['roc_times'], features['roc_values'], 'g-', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('ROC (intensity/hour)')
    ax2.set_title('Rate of Change', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Degradation Slope
    ax3 = fig.add_subplot(gs[1, 1])
    scatter = ax3.scatter(features['slope_times'], features['slope_values'],
                         c=features['r_squared'], cmap='viridis', s=20, alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Slope (intensity/hour)')
    ax3.set_title('Degradation Slope (colored by R²)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='R²')
    
    # 4. Sudden Shifts
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(times, smoothed, 'b-', linewidth=1, alpha=0.5)
    if len(features['shift_times']) > 0:
        shift_intensities = np.interp(features['shift_times'], times, smoothed)
        ax4.scatter(features['shift_times'], shift_intensities,
                   c='red', s=100, marker='v', 
                   label=f"Shifts (n={len(features['shift_times'])})", zorder=5)
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Sudden Shift Detection', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Statistics Table
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.axis('off')
    stats = features['statistics']
    stats_text = f"""
    Summary Statistics:
    
    Mean Intensity: {stats['mean_intensity']:.2f}
    Std Intensity: {stats['std_intensity']:.2f}
    Total Change: {stats['total_change']:.2f}
    
    Mean ROC: {stats['mean_roc']:.4f}
    Std ROC: {stats['std_roc']:.4f}
    
    Mean Slope: {stats['mean_slope']:.4f}
    
    Number of Shifts: {stats['num_shifts']}
    Duration: {stats['total_duration']:.1f} hours
    """
    ax5.text(0.1, 0.5, stats_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 6. Phase Space
    ax6 = fig.add_subplot(gs[3, 1])
    roc_interp = np.interp(times, features['roc_times'], features['roc_values'])
    scatter = ax6.scatter(intensities, roc_interp, c=times, cmap='plasma', s=15, alpha=0.6)
    ax6.set_xlabel('Intensity')
    ax6.set_ylabel('ROC')
    ax6.set_title('Phase Space', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Time (hours)')
    
    plt.suptitle(f'Comprehensive Feature Analysis - Indent {indent_id}',
                fontsize=14, fontweight='bold', y=0.995)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved analysis plot: {output_path}")
        plt.close()
    else:
        plt.show()
    
    return fig

# ==========================================
# MAIN PROCESSING PIPELINE
# ==========================================

def main():
    """Main processing pipeline for MP4 videos."""
    
    print("\n" + "="*70)
    print("CANTILEVER FATIGUE ANALYSIS - FEATURE EXTRACTION FROM MP4")
    print("="*70 + "\n")
    
    # Initialize extractors
    feature_extractor = FeatureExtractor()
    vector_encoder = VectorEncoder(n_bins=100)
    
    # Storage for all indent data
    global_indent_data = {}
    
    # Get reference from first video
    first_video_file = SEGMENT_CONFIG[0][0]
    first_video_path = os.path.join(VIDEO_DIR, first_video_file)
    
    if not os.path.exists(first_video_path):
        print(f"ERROR: Video file not found: {first_video_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory:")
        for f in os.listdir(VIDEO_DIR):
            print(f"  - {f}")
        sys.exit(1)
    
    # Get first frame for reference
    print(f"Opening {first_video_file} to establish reference...")
    cap = cv2.VideoCapture(first_video_path)
    ret, master_ref_img = cap.read()
    cap.release()
    
    if not ret:
        print("ERROR: Could not read first frame from video")
        sys.exit(1)
    
    master_ref_gray = cv2.cvtColor(master_ref_img, cv2.COLOR_BGR2GRAY)
    master_ref_stats = cv2.meanStdDev(master_ref_gray)
    print("✓ Global brightness reference established\n")
    
    # Process each video segment
    for seg_idx, segment_info in enumerate(SEGMENT_CONFIG):
        # Parse segment configuration
        if len(segment_info) == 5:
            video_file, period_start, period_end, skip_start_seconds, is_part2 = segment_info
        else:
            # Backward compatibility
            video_file, period_start, period_end, skip_start_seconds = segment_info
            is_part2 = False
        
        print(f"\n{'='*70}")
        print(f"SEGMENT: {video_file}")
        print(f"  Period: {period_start}-{period_end} hours")
        print(f"  Type: {'Part 2 (ends at period end)' if is_part2 else 'Part 1 (starts at period start)'}")
        if skip_start_seconds > 0:
            print(f"  Skipping first {skip_start_seconds} seconds ({skip_start_seconds/60:.1f} minutes)")
        print(f"{'='*70}")
        
        video_path = os.path.join(VIDEO_DIR, video_file)
        
        if not os.path.exists(video_path):
            print(f"  ⚠ Video file not found: {video_path}")
            continue
        
        # Get video info
        info = get_video_info(video_path)
        if info is None:
            print(f"  ⚠ Could not open video: {video_path}")
            continue
        
        print(f"  Video Properties:")
        print(f"    Total frames: {info['total_frames']:,}")
        print(f"    FPS: {info['fps']:.2f}")
        print(f"    Duration: {info['duration_hours']:.2f} hours")
        print(f"    Resolution: {info['width']}×{info['height']}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Skip to start position if specified
        if skip_start_seconds > 0:
            start_frame = int(skip_start_seconds * info['fps'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            print(f"  ✓ Skipped to frame {start_frame:,} ({skip_start_seconds/60:.1f} min)")
        
        # Get reference frame (after skip)
        ret, img_ref_raw = cap.read()
        if not ret:
            print(f"  ⚠ Could not read frame at skip position")
            cap.release()
            continue
        
        # For display: Use raw image (with color, before normalization)
        # For processing: Use normalized image
        img_ref_display = img_ref_raw.copy()  # Keep original for display
        img_ref = normalize_brightness(img_ref_raw, master_ref_stats)
        img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        
        # Get template
        template = load_or_create_template(img_ref_gray)
        if template is None:
            print("  ⚠ Template creation failed")
            cap.release()
            continue
        
        # Manual selection - SHOW RAW COLOR IMAGE
        targets = select_indents_manual(img_ref_display, template, title_prefix=video_file)
        if not targets:
            print(f"  ⚠ No indents selected, skipping...")
            cap.release()
            continue
        
        print(f"  ✓ Tracking {len(targets)} indents\n")
        
        # Initialize tracking
        active_indents = []
        for i, (tx, ty) in enumerate(targets):
            t_template = get_tracking_template(img_ref, tx, ty, size=40)
            indent_id = i + 1
            
            if indent_id not in global_indent_data:
                global_indent_data[indent_id] = {'times': [], 'intensities': []}
            
            active_indents.append({
                'id': indent_id,
                'current_x': tx,
                'current_y': ty,
                'template': t_template,
                'seg_times': [],
                'seg_intensities': []
            })
        
        # Calculate video duration for part2 timeline
        video_duration_hours = info['duration_hours'] - (skip_start_seconds / 3600)
        video_duration_seconds = video_duration_hours * 3600
        
        # Calculate sampling interval in frames
        if SAMPLING_INTERVAL_MINUTES > 0:
            frames_per_sample = int(SAMPLING_INTERVAL_MINUTES * 60 * info['fps'])
            print(f"  ⏱ Sampling: One frame every {SAMPLING_INTERVAL_MINUTES} minutes ({frames_per_sample:,} frames)")
            expected_samples = int(video_duration_hours * 60 / SAMPLING_INTERVAL_MINUTES)
            print(f"  📊 Expected samples: ~{expected_samples:,} over {video_duration_hours:.1f} hours")
        else:
            frames_per_sample = 1  # Process every frame
            expected_samples = info['total_frames'] - int(skip_start_seconds * info['fps'])
        
        # Timeline calculation info
        if is_part2:
            print(f"  🕐 Timeline: Counting backward from {period_end}h")
            print(f"     First sample: {period_end - video_duration_hours:.3f}h")
            print(f"     Last sample: {period_end:.3f}h")
        else:
            print(f"  🕐 Timeline: Counting forward from {period_start}h")  
            print(f"     First sample: {period_start:.3f}h")
            print(f"     Last sample: {period_start + video_duration_hours:.3f}h")
        
        # Process frames with temporal sampling
        processed_count = 0
        next_sample_frame = 0  # Next frame to sample (relative to skip position)
        
        while True:
            # Jump to next sample point
            current_frame_pos = int(skip_start_seconds * info['fps']) + next_sample_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate time within video (after skip)
            video_elapsed_seconds = next_sample_frame / info['fps']
            video_elapsed_hours = video_elapsed_seconds / 3600
            
            # Calculate experiment time based on part1/part2 logic
            if is_part2:
                # Part 2: Count backward from period end
                # experiment_time = period_end - video_duration + video_elapsed
                time_hours = period_end - video_duration_hours + video_elapsed_hours
            else:
                # Part 1: Count forward from period start
                # experiment_time = period_start + video_elapsed
                time_hours = period_start + video_elapsed_hours
            
            # Normalize and align
            img = normalize_brightness(frame, master_ref_stats)
            
            # Alignment (skip for first sample)
            if processed_count == 0:
                aligned = img
            else:
                aligned = align_image_ecc(img, img_ref)
                if aligned is None:
                    aligned = img  # Use unaligned if alignment fails
            
            # Track each indent
            for indent in active_indents:
                cx, cy, conf = track_template_local(
                    aligned, indent['current_x'], indent['current_y'],
                    indent['template'], search_window=80
                )
                
                if conf <= 0.6:
                    cx, cy, conf = track_template_local(
                        aligned, indent['current_x'], indent['current_y'],
                        indent['template'], search_window=120
                    )
                    if conf <= 0.5:
                        cx, cy = indent['current_x'], indent['current_y']
                
                # Update template periodically (every 50 samples)
                if processed_count % 50 == 0 and conf > 0.8:
                    indent['template'] = get_tracking_template(aligned, cx, cy, size=40)
                
                indent['current_x'] = cx
                indent['current_y'] = cy
                
                intensity = get_indent_intensity(aligned, cx, cy, radius=8)
                
                indent['seg_times'].append(time_hours)
                indent['seg_intensities'].append(intensity)
            
            processed_count += 1
            
            # Progress update
            if processed_count % 50 == 0 or processed_count == expected_samples:
                progress = (processed_count / expected_samples) * 100 if expected_samples > 0 else 0
                if is_part2:
                    elapsed_from_period_start = time_hours - (period_end - video_duration_hours)
                else:
                    elapsed_from_period_start = time_hours - period_start
                print(f"\r  Processing: {processed_count:,}/{expected_samples:,} samples ({progress:.1f}%) | Time: {time_hours:.2f}h", end="")
            
            # Move to next sample point
            next_sample_frame += frames_per_sample
        
        print(f"\n  ✓ Processed {processed_count:,} frames")
        cap.release()
        
        # Merge to global data
        for indent in active_indents:
            gid = indent['id']
            global_indent_data[gid]['times'].extend(indent['seg_times'])
            global_indent_data[gid]['intensities'].extend(indent['seg_intensities'])
    
    # ==========================================
    # FEATURE EXTRACTION & VECTOR GENERATION
    # ==========================================
    
    print(f"\n{'='*70}")
    print("EXTRACTING FEATURES & GENERATING VECTORS")
    print(f"{'='*70}\n")
    
    all_features = {}
    all_vectors = {}
    all_binned_features = {}
    
    for indent_id, data in global_indent_data.items():
        if len(data['times']) < 20:
            print(f"  ⚠ Indent {indent_id}: Insufficient data ({len(data['times'])} points)")
            continue
        
        print(f"  Processing Indent {indent_id}...")
        
        # Extract features
        features = feature_extractor.extract_all_features(data['times'], data['intensities'])
        all_features[indent_id] = features
        
        # Generate vectors
        stat_vector, stat_labels = vector_encoder.statistical_summary_vector(features)
        all_vectors[indent_id] = {'statistical': stat_vector, 'labels': stat_labels}
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  USER CHANGE (optional): Change (0, 72) if your total          │
        # │  experiment duration is not 72 hours.                          │
        # └─────────────────────────────────────────────────────────────────┘
        time_range = (0, 72)
        binned_features, bin_centers = vector_encoder.time_binned_features(features, time_range=time_range)
        all_binned_features[indent_id] = {'features': binned_features, 'bin_centers': bin_centers}
        
        # Generate plots
        plot_path = f"indent_{indent_id}_analysis.png"
        plot_comprehensive_analysis(indent_id, features, output_path=plot_path)
        
        print(f"    ✓ Extracted {len(features['shift_times'])} shifts")
        print(f"    ✓ Mean ROC: {features['statistics']['mean_roc']:.4f}")
        print(f"    ✓ Mean slope: {features['statistics']['mean_slope']:.4f}")
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    # Save as NumPy arrays
    for indent_id in all_vectors.keys():
        np.save(f"indent_{indent_id}_statistical_vector.npy", all_vectors[indent_id]['statistical'])
        np.save(f"indent_{indent_id}_binned_features.npy", all_binned_features[indent_id]['features'])
        print(f"  ✓ Saved vectors for Indent {indent_id}")
    
    # Save as CSV
    for indent_id, features in all_features.items():
        df = pd.DataFrame({
            'time_hours': features['times'],
            'intensity': features['intensities'],
            'roc': np.interp(features['times'], features['roc_times'], features['roc_values'])
        })
        df.to_csv(f"indent_{indent_id}_timeseries.csv", index=False)
        print(f"  ✓ Saved time series CSV for Indent {indent_id}")
    
    # Save metadata
    metadata = {
        'feature_config': FEATURE_CONFIG,
        'sampling_interval_minutes': SAMPLING_INTERVAL_MINUTES,
        'segment_config': [
            {
                'video_file': s[0], 
                'period_start': s[1], 
                'period_end': s[2],
                'skip_start_seconds': s[3] if len(s) > 3 else 0,
                'is_part2': s[4] if len(s) > 4 else False,
                'timeline_type': 'backward_from_end' if (len(s) > 4 and s[4]) else 'forward_from_start'
            }
            for s in SEGMENT_CONFIG
        ],
        'csv_mapping': CSV_VIDEO_MAPPING,
        'indents': {}
    }
    
    for indent_id, features in all_features.items():
        metadata['indents'][indent_id] = {
            'num_points': len(features['times']),
            'time_range': [float(features['times'][0]), float(features['times'][-1])],
            'statistics': {k: float(v) for k, v in features['statistics'].items()},
            'num_shifts': len(features['shift_times']),
            'shift_times': [float(t) for t in features['shift_times']]
        }
    
    with open('analysis_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Saved metadata JSON")
    
    # Create summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nProcessed {len(all_features)} indents")
    print(f"Total shifts detected: {sum(len(f['shift_times']) for f in all_features.values())}")
    print(f"\nOutputs:")
    print(f"  - Individual analysis plots (indent_N_analysis.png)")
    print(f"  - Statistical vectors (indent_N_statistical_vector.npy)")
    print(f"  - Time-binned feature arrays (indent_N_binned_features.npy)")
    print(f"  - Time series CSV files (indent_N_timeseries.csv)")
    print(f"  - Metadata JSON (analysis_metadata.json)")
    print(f"\nReady for correlation with accelerometer data!")

if __name__ == "__main__":
    main()
