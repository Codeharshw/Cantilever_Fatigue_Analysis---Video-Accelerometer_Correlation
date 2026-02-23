# Cantilever Fatigue Analysis — Video + Accelerometer Correlation

A research-grade pipeline for tracking microstructural degradation in cantilever fatigue experiments using optical microscopy video and IMU accelerometer data.

---

## Experiment Context

During a **72-hour continuous cantilever fatigue run**, microscopy videos were recorded to track indent features on the cantilever surface while an IMU (accelerometer + gyroscope) simultaneously logged vibration and bending data.

- **Setup**: A cantilever beam was run for 72 hours under continuous cyclic loading
- **Video**: Optical microscopy videos captured surface indents throughout the run, recorded at 10fps
- **IMU**: An accelerometer/gyroscope logged bending acceleration, lateral acceleration, angular velocity, and temperature at high frequency
- **Goal**: Correlate what we see optically (indent intensity changes) with what the accelerometer measures (vibration/fatigue loading) to build a multi-sensor fatigue model

---

## Why the video config handles 6 segments instead of 3

Ideally, one video file per 24-hour period would be cleanest. In practice, the original experiment had **camera interruptions mid-period** — the recording had to be stopped and restarted, producing a "part 1" (start of period) and "part 2" (end of period) for each day, with a gap in the middle where data was lost.

This is why `fatigue_feature_extraction_from_mp4.py` supports a `is_part2` flag and backward timeline counting — it reconstructs where in the experiment each video fragment sits, even when the middle portion is missing. If your recording is continuous (one file per period), just set `is_part2=False` and `skip_start_seconds=0` for all entries and it works identically to a simple 3-video setup.

---

## Files

| File | Purpose |
|---|---|
| `fatigue_feature_extraction_from_mp4.py` | Step 1 — Extract indent intensity features from MP4 videos |
| `accelerometer_video_correlation.py` | Step 3 — Correlate video features with accelerometer data |
| `roc_nan_fix.py` | Step 2 (if needed) — Diagnose and fix NaN values in ROC data |
| `json_to_csv.py` | Step 4 — Convert correlation JSON summary to a matrix CSV |

---

## Pipeline Order

```
Step 1:  fatigue_feature_extraction_from_mp4.py
         → outputs: indent_N_timeseries.csv, analysis_metadata.json, plots

Step 2:  roc_nan_fix.py           (only if ROC column contains NaN)
         → outputs: indent_N_timeseries_fixed.csv
         → rename these to replace the originals before Step 3

Step 3:  accelerometer_video_correlation.py
         → outputs: indent_N_correlations.json, correlation plots

Step 4:  json_to_csv.py
         → outputs: final_summary_matrix.csv
```

---

## Requirements

```
opencv-python
numpy
scipy
matplotlib
pandas
scikit-learn
seaborn
```

Install with:
```bash
pip install opencv-python numpy scipy matplotlib pandas scikit-learn seaborn
```

---

## Known Issues

- **ROC NaN at segment boundaries**: When videos from different periods are stitched together, the rate-of-change (ROC) at the 24h and 48h boundary points becomes NaN because the gradient is computed across a time discontinuity. Run `roc_nan_fix.py` to detect and repair this automatically.
- **Template file**: On first run you will be prompted to draw a box around one indent region. This is saved as `indent_template_1.png` and reused on all subsequent runs. Delete this file if you want to reselect the region or upload your own cropped indent separately, saving it as `indent_template_1.png`.
- **ECC alignment failures**: Image alignment occasionally fails on very low-contrast frames. The script falls back to the unaligned frame — this is expected and printed as a warning.
- **Accelerometer time sync**: The CSV timestamps are assumed to start at 0 for each recording period (i.e., second 0 of the CSV = `period_start` of that period). If your accelerometer clock does not reset between periods, adjust the `experiment_time` calculation in the loader.

---

## License

Research use. Please cite appropriately if you use this pipeline.
