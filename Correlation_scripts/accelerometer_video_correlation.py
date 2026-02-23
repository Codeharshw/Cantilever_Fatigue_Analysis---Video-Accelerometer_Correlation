"""
ACCELEROMETER-VIDEO CORRELATION WITH 2-MINUTE WINDOWING
========================================================

Correlates video indent features with accelerometer measurements using
2-minute time windows for proper multi-sensor fusion.

Author: Research-grade implementation  
Date: 2026-02-14
Modified: 2026-02-16 - Updated CSV filenames
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.interpolate import interp1d
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE (optional): Set this to the folder containing your        │
# │  accelerometer CSV files if they are not in the same directory as      │
# │  this script.                                                          │
# └─────────────────────────────────────────────────────────────────────────┘
CSV_DIR = "."  # Current directory

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE REQUIRED: Update these to match the actual column names   │
# │  in your accelerometer CSV files. Open one CSV and check the headers.  │
# │                                                                        │
# │  'time': timestamp column in SECONDS from the start of each recording.│
# │  'ax/ay/az': linear acceleration on each axis (in g or m/s²).         │
# │  'gx/gy/gz': angular velocity on each axis (gyroscope).               │
# │  'temp': temperature column.                                           │
# └─────────────────────────────────────────────────────────────────────────┘
ACCEL_COLUMNS = {
    'time': 'PC_Timestamp',
    'ax': 'ax',  # Linear acceleration X
    'ay': 'ay',  # Linear acceleration Y  
    'az': 'az',  # Linear acceleration Z
    'gx': 'gx',  # Angular velocity X (gyro)
    'gy': 'gy',  # Angular velocity Y
    'gz': 'gz',  # Angular velocity Z
    'temp': 'Temperature_C'
}

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE REQUIRED: Set 'bending_axis' to the accelerometer axis   │
# │  that corresponds to your cantilever's primary bending direction.      │
# │                                                                        │
# │  For a cantilever bending up/down: typically 'az' (vertical).         │
# │  Update based on how your sensor is physically mounted.               │
# │                                                                        │
# │  'gravity_calibrated': True if your IMU firmware already subtracts    │
# │  gravity (all axes read ~0 at rest). False if az reads ~±1g at rest.  │
# └─────────────────────────────────────────────────────────────────────────┘
SENSOR_CONFIG = {
    'bending_axis': 'az',     # PRIMARY FATIGUE AXIS: Up-down (vertical bending)
    'length_axis': 'ax',      # Axis along cantilever length
    'lateral_axis': 'ay',     # Lateral (side-to-side) axis
    'accel_units': 'g',       # 'g' or 'm/s²'
    'gyro_units': 'deg/s',    # 'deg/s' or 'rad/s'
    'gravity_calibrated': True,  # All axes calibrated to zero (no gravity component)
}

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  USER CHANGE (optional): Tune the windowing parameters.               │
# │                                                                        │
# │  window_minutes: total window width centred on each video sample.     │
# │    Should match SAMPLING_INTERVAL_MINUTES from the extraction script  │
# │    so each video sample aligns with one accelerometer window.         │
# │  min_samples: minimum accelerometer rows needed in a window. Windows  │
# │    with fewer are flagged as warnings and produce zero features.       │
# └─────────────────────────────────────────────────────────────────────────┘
WINDOW_CONFIG = {
    'window_minutes': 2.0,    # ±1 minute around each video sample
    'overlap': 0.5,           # 50% overlap for continuous analysis
    'min_samples': 10,        # Minimum accelerometer samples in window
}

# ==========================================
# ACCELEROMETER DATA LOADING
# ==========================================

class AccelerometerDataLoader:
    """Load and preprocess accelerometer CSV files."""
    
    def __init__(self, csv_mapping, csv_dir=CSV_DIR):
        """
        Args:
            csv_mapping: Dict from video processing metadata
            csv_dir: Directory containing CSV files
        """
        self.csv_mapping = csv_mapping
        self.csv_dir = csv_dir
        self.data = {}
        
    def load_all_csvs(self):
        """Load all accelerometer CSV files."""
        for csv_file, info in self.csv_mapping.items():
            csv_path = os.path.join(self.csv_dir, csv_file)
            
            if not os.path.exists(csv_path):
                print(f"⚠ Warning: {csv_file} not found, skipping")
                continue
            
            print(f"Loading {csv_file}...")
            df = pd.read_csv(csv_path)
            
            # Convert time to hours (PC_Timestamp is already in seconds)
            df['time_hours'] = df[ACCEL_COLUMNS['time']] / 3600.0
            
            # Add period information
            df['period_start'] = info['period_start']
            df['period_end'] = info['period_end']
            
            # ┌──────────────────────────────────────────────────────────────┐
            # │  NOTE: This assumes the CSV timestamp resets to 0 at the    │
            # │  start of each recording period. If your accelerometer clock │
            # │  does not reset between periods, adjust this line so that   │
            # │  experiment_time correctly maps to the 0-72h timeline.      │
            # └──────────────────────────────────────────────────────────────┘
            # Calculate absolute experiment time
            # Assuming CSV time starts at period_start
            df['experiment_time'] = info['period_start'] + df['time_hours']
            
            self.data[csv_file] = {
                'dataframe': df,
                'period': (info['period_start'], info['period_end']),
                'videos': info['videos']
            }
            
            print(f"  ✓ Loaded {len(df):,} samples")
            print(f"  ✓ Time range: {df['experiment_time'].min():.2f}h - {df['experiment_time'].max():.2f}h")
        
        return self.data
    
    def get_data_for_time(self, experiment_time_hours, window_minutes=2.0):
        """
        Get accelerometer data in a time window around a specific time.
        
        Args:
            experiment_time_hours: Center time in experiment timeline
            window_minutes: Window size in minutes (total window is ±window_minutes/2)
            
        Returns:
            DataFrame with accelerometer data in the window
        """
        window_hours = window_minutes / 60.0
        t_start = experiment_time_hours - window_hours / 2
        t_end = experiment_time_hours + window_hours / 2
        
        # Find which CSV period this time belongs to
        for csv_file, info in self.data.items():
            period_start, period_end = info['period']
            
            if period_start <= experiment_time_hours <= period_end:
                df = info['dataframe']
                
                # Extract window
                mask = (df['experiment_time'] >= t_start) & (df['experiment_time'] <= t_end)
                window_data = df[mask].copy()
                
                if len(window_data) < WINDOW_CONFIG['min_samples']:
                    print(f"  ⚠ Warning: Only {len(window_data)} samples in window at t={experiment_time_hours:.2f}h")
                
                return window_data
        
        print(f"  ⚠ Warning: No accelerometer data found for t={experiment_time_hours:.2f}h")
        return None

# ==========================================
# FEATURE EXTRACTION FROM ACCELEROMETER
# ==========================================

class AccelFeatureExtractor:
    """Extract features from accelerometer time windows."""
    
    def __init__(self, sensor_config=SENSOR_CONFIG):
        self.config = sensor_config
        
    def compute_features(self, accel_window):
        """
        Compute all features from accelerometer window.
        
        Args:
            accel_window: DataFrame with accelerometer data
            
        Returns:
            Dictionary of features
        """
        if accel_window is None or len(accel_window) == 0:
            return self._empty_features()
        
        features = {}
        
        # 1. Bending Acceleration Features
        bending_axis = self.config['bending_axis']
        bending_accel = accel_window[bending_axis].values
        
        features['bending_rms'] = self._rms(bending_accel)
        features['bending_mean'] = np.mean(np.abs(bending_accel))
        features['bending_peak'] = np.max(np.abs(bending_accel))
        features['bending_std'] = np.std(bending_accel)
        
        # 2. Total Acceleration (3D magnitude)
        ax = accel_window[ACCEL_COLUMNS['ax']].values
        ay = accel_window[ACCEL_COLUMNS['ay']].values
        az = accel_window[ACCEL_COLUMNS['az']].values
        
        # All axes already calibrated to zero (no gravity component)
        # So we can directly compute magnitude without gravity correction
        total_accel = np.sqrt(ax**2 + ay**2 + az**2)
        
        features['total_rms'] = self._rms(total_accel)
        features['total_mean'] = np.mean(total_accel)
        features['total_peak'] = np.max(total_accel)
        
        # 3. Angular Velocity Features
        gx = accel_window[ACCEL_COLUMNS['gx']].values
        gy = accel_window[ACCEL_COLUMNS['gy']].values
        gz = accel_window[ACCEL_COLUMNS['gz']].values
        
        total_gyro = np.sqrt(gx**2 + gy**2 + gz**2)
        
        features['gyro_rms'] = self._rms(total_gyro)
        features['gyro_mean'] = np.mean(total_gyro)
        features['gyro_peak'] = np.max(total_gyro)
        
        # 4. Vibration Energy (proxy)
        dt = np.diff(accel_window['time_hours'].values * 3600)  # Convert to seconds
        if len(dt) > 0:
            avg_dt = np.mean(dt)
            # Energy = integral of acceleration² over time
            features['vibration_energy'] = np.sum(total_accel**2) * avg_dt
        else:
            features['vibration_energy'] = 0.0
        
        # 5. Strain Proxy (cumulative energy)
        features['cumulative_strain_proxy'] = np.sum(np.abs(bending_accel))
        
        # 6. Impact/Spike Detection
        # Count number of samples exceeding 3σ
        threshold = np.mean(total_accel) + 3 * np.std(total_accel)
        spikes = total_accel > threshold
        features['num_spikes'] = np.sum(spikes)
        features['spike_rate'] = features['num_spikes'] / len(total_accel) if len(total_accel) > 0 else 0
        
        # 7. Frequency Content (if enough samples)
        if len(total_accel) > 50:
            try:
                # Estimate dominant frequency using FFT
                fft_vals = np.fft.fft(total_accel - np.mean(total_accel))
                fft_freq = np.fft.fftfreq(len(total_accel), d=avg_dt)
                
                # Get positive frequencies only
                pos_mask = fft_freq > 0
                fft_magnitude = np.abs(fft_vals[pos_mask])
                fft_freq_pos = fft_freq[pos_mask]
                
                # Find dominant frequency
                if len(fft_magnitude) > 0:
                    dominant_idx = np.argmax(fft_magnitude)
                    features['dominant_frequency_hz'] = fft_freq_pos[dominant_idx]
                else:
                    features['dominant_frequency_hz'] = 0.0
            except:
                features['dominant_frequency_hz'] = 0.0
        else:
            features['dominant_frequency_hz'] = 0.0
        
        # 8. Temperature (might affect material properties)
        features['temperature'] = np.mean(accel_window[ACCEL_COLUMNS['temp']].values)
        
        return features
    
    def _rms(self, signal):
        """Root mean square."""
        return np.sqrt(np.mean(signal**2))
    
    def _empty_features(self):
        """Return zero features when no data available."""
        return {
            'bending_rms': 0.0,
            'bending_mean': 0.0,
            'bending_peak': 0.0,
            'bending_std': 0.0,
            'total_rms': 0.0,
            'total_mean': 0.0,
            'total_peak': 0.0,
            'gyro_rms': 0.0,
            'gyro_mean': 0.0,
            'gyro_peak': 0.0,
            'vibration_energy': 0.0,
            'cumulative_strain_proxy': 0.0,
            'num_spikes': 0,
            'spike_rate': 0.0,
            'dominant_frequency_hz': 0.0,
            'temperature': 0.0
        }

# ==========================================
# CORRELATION ANALYSIS
# ==========================================

class VideoAccelCorrelator:
    """Correlate video indent features with accelerometer features."""
    
    def __init__(self, metadata_path, csv_dir=CSV_DIR):
        """
        Args:
            metadata_path: Path to analysis_metadata.json from video processing
            csv_dir: Directory containing accelerometer CSV files
        """
        # Load video metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load accelerometer data
        self.accel_loader = AccelerometerDataLoader(
            self.metadata['csv_mapping'], 
            csv_dir
        )
        self.accel_data = self.accel_loader.load_all_csvs()
        
        # Initialize feature extractor
        self.accel_extractor = AccelFeatureExtractor()
        
    def correlate_indent(self, indent_id, output_dir='.'):
        """
        Full correlation analysis for one indent.
        
        Args:
            indent_id: Indent number (1, 2, 3, ...)
            output_dir: Where to save outputs
        """
        print(f"\n{'='*70}")
        print(f"CORRELATING INDENT {indent_id} WITH ACCELEROMETER")
        print(f"{'='*70}\n")
        
        # Load video time series
        ts_path = f'indent_{indent_id}_timeseries.csv'
        if not os.path.exists(ts_path):
            print(f"⚠ Error: {ts_path} not found")
            return None
        
        video_ts = pd.read_csv(ts_path)
        print(f"✓ Loaded video data: {len(video_ts)} time points")
        
        # Extract accelerometer features at each video time point
        print(f"Extracting accelerometer features...")
        
        accel_features_list = []
        valid_indices = []
        
        for idx, row in video_ts.iterrows():
            time_hours = row['time_hours']
            
            # Get accelerometer window
            accel_window = self.accel_loader.get_data_for_time(
                time_hours, 
                window_minutes=WINDOW_CONFIG['window_minutes']
            )
            
            # Extract features
            features = self.accel_extractor.compute_features(accel_window)
            features['video_time'] = time_hours
            features['video_intensity'] = row['intensity']
            features['video_roc'] = row['roc']
            
            accel_features_list.append(features)
            
            if accel_window is not None and len(accel_window) >= WINDOW_CONFIG['min_samples']:
                valid_indices.append(idx)
            
            if idx % 50 == 0:
                print(f"\r  Processed {idx+1}/{len(video_ts)} time points", end="")
        
        print(f"\n✓ Extracted features for {len(valid_indices)} valid windows")
        
        # Convert to DataFrame
        corr_df = pd.DataFrame(accel_features_list)
        
        # Save combined dataset
        output_path = os.path.join(output_dir, f'indent_{indent_id}_video_accel_combined.csv')
        corr_df.to_csv(output_path, index=False)
        print(f"✓ Saved combined data: {output_path}")
        
        # Compute correlations
        print(f"\nComputing correlations...")
        correlations = self._compute_correlations(corr_df)
        
        # Save correlation results
        corr_json_path = os.path.join(output_dir, f'indent_{indent_id}_correlations.json')
        with open(corr_json_path, 'w') as f:
            json.dump(correlations, f, indent=2)
        print(f"✓ Saved correlations: {corr_json_path}")
        
        # Generate plots
        print(f"\nGenerating correlation plots...")
        self._plot_correlations(corr_df, indent_id, output_dir)
        
        return correlations
    
    def _compute_correlations(self, df):
        """Compute correlation metrics between video and accel features."""
        
        # Filter out rows with missing data
        valid_df = df[df['bending_rms'] > 0].copy()
        
        if len(valid_df) < 10:
            print("⚠ Warning: Too few valid data points for correlation")
            return {}
        
        correlations = {}
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  USER CHANGE (optional): Add or remove feature pairs to control │
        # │  which combinations are computed and saved to the JSON output.  │
        # │  Format: ('video_feature', 'accel_feature', 'Label')           │
        # └─────────────────────────────────────────────────────────────────┘
        pairs = [
            ('video_intensity', 'bending_rms', 'Intensity vs Bending RMS'),
            ('video_intensity', 'total_rms', 'Intensity vs Total RMS'),
            ('video_intensity', 'vibration_energy', 'Intensity vs Vibration Energy'),
            ('video_roc', 'bending_rms', 'ROC vs Bending RMS'),
            ('video_roc', 'vibration_energy', 'ROC vs Vibration Energy'),
            ('video_roc', 'gyro_rms', 'ROC vs Angular Velocity'),
        ]
        
        for video_feat, accel_feat, label in pairs:
            if video_feat in valid_df.columns and accel_feat in valid_df.columns:
                # Pearson correlation
                r, p = stats.pearsonr(valid_df[video_feat], valid_df[accel_feat])
                
                # Spearman correlation (for non-linear)
                rho, p_spear = stats.spearmanr(valid_df[video_feat], valid_df[accel_feat])
                
                correlations[label] = {
                    'pearson_r': float(r),
                    'pearson_p': float(p),
                    'spearman_rho': float(rho),
                    'spearman_p': float(p_spear),
                    'n_samples': len(valid_df)
                }
        
        return correlations
    
    def _plot_correlations(self, df, indent_id, output_dir):
        """Generate correlation visualization plots."""
        
        valid_df = df[df['bending_rms'] > 0].copy()
        
        if len(valid_df) < 10:
            print("⚠ Not enough data for plotting")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Time series overlay
        ax1 = fig.add_subplot(gs[0, :])
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(valid_df['video_time'], valid_df['video_intensity'],
                        'b-', linewidth=1.5, alpha=0.7, label='Video Intensity')
        line2 = ax1_twin.plot(valid_df['video_time'], valid_df['bending_rms'],
                             'r-', linewidth=1.5, alpha=0.7, label='Bending RMS')
        
        ax1.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Video Intensity', color='b', fontsize=11)
        ax1_twin.set_ylabel('Bending RMS', color='r', fontsize=11)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        ax1.set_title(f'Indent {indent_id}: Time Series Comparison', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # Plot 2: Intensity vs Bending RMS
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(valid_df['bending_rms'], valid_df['video_intensity'],
                   alpha=0.6, s=30, c=valid_df['video_time'], cmap='viridis')
        r, p = stats.pearsonr(valid_df['bending_rms'], valid_df['video_intensity'])
        ax2.set_xlabel('Bending RMS', fontsize=10)
        ax2.set_ylabel('Video Intensity', fontsize=10)
        ax2.set_title(f'Intensity vs Bending RMS (r={r:.3f}, p={p:.2e})', 
                     fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: ROC vs Vibration Energy
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(valid_df['vibration_energy'], valid_df['video_roc'],
                   alpha=0.6, s=30, c=valid_df['video_time'], cmap='viridis')
        r, p = stats.pearsonr(valid_df['vibration_energy'], valid_df['video_roc'])
        ax3.set_xlabel('Vibration Energy', fontsize=10)
        ax3.set_ylabel('Video ROC', fontsize=10)
        ax3.set_title(f'ROC vs Vibration Energy (r={r:.3f}, p={p:.2e})',
                     fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Intensity vs Total RMS
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.scatter(valid_df['total_rms'], valid_df['video_intensity'],
                   alpha=0.6, s=30, c=valid_df['video_time'], cmap='viridis')
        r, p = stats.pearsonr(valid_df['total_rms'], valid_df['video_intensity'])
        ax4.set_xlabel('Total RMS Accel', fontsize=10)
        ax4.set_ylabel('Video Intensity', fontsize=10)
        ax4.set_title(f'Intensity vs Total RMS (r={r:.3f}, p={p:.2e})',
                     fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Correlation heatmap
        ax5 = fig.add_subplot(gs[2, :])
        
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  USER CHANGE (optional): Edit this list to add or remove       │
        # │  features from the correlation heatmap.                        │
        # └─────────────────────────────────────────────────────────────────┘
        heatmap_features = ['video_intensity', 'video_roc', 
                           'bending_rms', 'total_rms', 'vibration_energy',
                           'gyro_rms', 'bending_peak', 'temperature']
        
        available_features = [f for f in heatmap_features if f in valid_df.columns]
        corr_matrix = valid_df[available_features].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, ax=ax5,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax5.set_title('Feature Correlation Matrix', 
                     fontsize=11, fontweight='bold')
        
        plt.suptitle(f'Video-Accelerometer Correlation Analysis - Indent {indent_id}',
                    fontsize=14, fontweight='bold', y=0.995)
        
        output_path = os.path.join(output_dir, f'indent_{indent_id}_correlation_plots.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved correlation plots: {output_path}")

# ==========================================
# BATCH PROCESSING
# ==========================================

def correlate_all_indents(metadata_path='analysis_metadata.json', 
                          csv_dir='.', 
                          output_dir='.'):
    """
    Process all indents and correlate with accelerometer data.
    
    Args:
        metadata_path: Path to video analysis metadata
        csv_dir: Directory with accelerometer CSV files
        output_dir: Where to save outputs
    """
    print("\n" + "="*70)
    print("VIDEO-ACCELEROMETER CORRELATION - BATCH PROCESSING")
    print("="*70 + "\n")
    
    # Initialize correlator
    correlator = VideoAccelCorrelator(metadata_path, csv_dir)
    
    # Get list of indents from metadata
    indent_ids = list(correlator.metadata['indents'].keys())
    print(f"Found {len(indent_ids)} indents to process\n")
    
    # Process each indent
    all_results = {}
    for indent_id in indent_ids:
        try:
            results = correlator.correlate_indent(int(indent_id), output_dir)
            all_results[indent_id] = results
        except Exception as e:
            print(f"⚠ Error processing indent {indent_id}: {e}")
            continue
    
    # Save summary
    summary_path = os.path.join(output_dir, 'correlation_summary_all_indents.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Processed {len(all_results)} indents")
    print(f"Summary saved: {summary_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("\nACCELEROMETER-VIDEO CORRELATION ANALYSIS")
    print("="*70)
    print("\nThis script correlates video indent features with accelerometer data")
    print("using 2-minute time windows.\n")
    
    if not os.path.exists('analysis_metadata.json'):
        print("⚠ Error: analysis_metadata.json not found")
        print("Please run fatigue_feature_extraction_from_mp4.py first")
    else:
        # ┌─────────────────────────────────────────────────────────────────┐
        # │  USER CHANGE (optional): Update csv_dir or output_dir if your  │
        # │  CSV files or desired output folder differ from the current dir.│
        # └─────────────────────────────────────────────────────────────────┘
        correlate_all_indents(
            metadata_path='analysis_metadata.json',
            csv_dir='.',
            output_dir='.'
        )
